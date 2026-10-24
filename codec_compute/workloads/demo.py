"""Compile useful image tasks across targets and decode the saved bitstreams.

This is a capability/correctness survey, not a runtime benchmark. Unsupported
semantics and bounded search misses remain distinct, explicit report entries.
No task changes its arithmetic or borders to make a target accept it.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import codec_compute as media
from codec_compute.synthesis.lower import SynthesisError
from decoder_core.decode import available, decode as decode_file
from .kernels import WORKLOADS, fixtures, reference


ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("h264", "hevc", "av1")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def save_plane(root, relative, board, bounds):
    pixels = bytes(value for row in board for value in row)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pixels)
    return {"path": relative, "sha256": digest(pixels), "bytes": len(pixels),
            "width": len(board[0]), "height": len(board), "range": list(bounds)}


def decode_saved(program, path, decoder):
    """Read the exact artifact on disk, without re-encoding its input pictures."""
    if program.target == "h264":
        from codec_compute.targets.h264 import _legacy
        raw = decode_file(path, decoder)
        return _legacy().backend.read(raw, program.native), raw, "guarded I420, including chroma"
    if program.target == "hevc":
        from codec_compute.hevc import decode_frames
        frames = decode_frames(path.read_bytes(), program.graph.width, program.graph.height)
        manifest = program.manifest()
        if len(frames) != manifest["costs"]["total_pictures"]:
            raise AssertionError("HEVC decoded picture count differs from compiled program")
        width, height = program.graph.width, program.graph.height
        return ({name: [list(frames[index][y*width:(y+1)*width]) for y in range(height)]
                 for name, index in manifest["output_frames"].items()}, b"".join(frames), "displayed luma planes")
    native = program.native.program
    result = native.decode(path.read_bytes(), decoder=decoder)
    outputs = {name: result.frames[value] for name, value in program.native.output_names.items()}
    raw = bytes(value for name in native.frame_names for row in result.frames[name] for value in row)
    return outputs, raw, "displayed monochrome planes"


def run(output, *, targets=TARGETS, tasks=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = [task for task in WORKLOADS if tasks is None or task.name in tasks]
    if not selected or set(targets) - set(TARGETS) or not targets:
        raise ValueError("Select known nonempty tasks and targets")
    if tasks is not None and set(tasks) - {task.name for task in WORKLOADS}:
        raise ValueError("Unknown task")
    decoders = {"h264": [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)],
                "hevc": ["ffmpeg"], "av1": ["libdav1d", "libaom-av1"]}
    if not available("ffmpeg"):
        raise RuntimeError("FFmpeg is required")
    serializer = None
    if "av1" in targets:
        from codec_compute.av1 import bootstrap
        executable = bootstrap.bridge_path()  # Never bootstrap implicitly.
        serializer = {"binary_sha256": digest(executable.read_bytes()),
                      "libaom_revision": bootstrap.REVISION}
    source_paths = sorted(path for path in (ROOT / "codec_compute").rglob("*.py")
                          if "artifacts" not in path.parts and "tests" not in path.parts)
    source_paths += [ROOT / "decoder_core" / "bitstream.py", ROOT / "decoder_core" / "decode.py"]
    source_paths += [ROOT / "codec_compute" / "av1" / name for name in ("bridge.c", "libaom.patch")]
    source_paths += sorted((ROOT / "05-h264-compiler" / "h264").glob("*.py"))
    sources = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in source_paths}
    cases = []
    for task in selected:
        graph = media.trace(task.kernel, width=32, height=32, input_ranges=task.input_ranges)
        for target in targets:
            entry = {"task": task.name, "target": target, "description": task.description,
                     "border": task.border}
            try:
                program = media.compile_graph(graph, target=target)
            except SynthesisError as error:
                entry.update(status="search_miss", reason=str(error), operation=error.operation,
                             search_status=error.result.status, search=error.result.manifest())
            except media.UnsupportedOperation as error:
                entry.update(status="unsupported", reason=str(error), operation=error.operation)
            else:
                manifest, tail = program.manifest(), program.tail
                coded_height, coded_width = manifest["coded_shape"]
                variants = []
                for variant in range(2):
                    inputs = fixtures(task, variant)
                    expected = reference(task, inputs)
                    if program.reference(inputs) != expected:
                        raise AssertionError(task.name + ": semantic IR disagrees with independent task formula")
                    stem = f"{task.name}-{target}-{variant}"
                    path = program.save(output / "streams" / (stem + program.native.suffix), inputs)
                    stream = path.read_bytes()
                    if program.tail != tail or not stream.endswith(tail):
                        raise AssertionError("Input-dependent computation tail")
                    checks, full_raw, actual = [], None, None
                    for decoder in decoders[target]:
                        actual, raw, format_ = decode_saved(program, path, decoder)
                        if actual != expected:
                            raise AssertionError(f"{stem}/{decoder}: decoder differs from independent task formula")
                        if full_raw is not None and raw != full_raw:
                            raise AssertionError(f"{stem}/{decoder}: full decoded planes disagree")
                        full_raw = raw
                        checks.append({"decoder": decoder, "status": "pass", "full_decoded_sha256": digest(raw),
                                       "decoded_bytes": len(raw), "decoded_format": format_})
                    ranges = {name: (node.lo, node.hi) for name, node in graph.outputs}
                    record = {"index": variant, "stream": str(path.relative_to(output)),
                              "stream_sha256": digest(stream), "stream_bytes": len(stream),
                              "manifest": str(path.with_suffix(".json").relative_to(output)),
                              "manifest_sha256": digest(path.with_suffix(".json").read_bytes()),
                              "input_prefix_bytes": len(stream)-len(tail),
                              "fixed_tail_verified": True, "decoders": checks,
                              "inputs": {name: save_plane(output, f"planes/{stem}/input-{name}.u8", board,
                                                          task.input_ranges[name]) for name, board in inputs.items()},
                              "outputs": {name: save_plane(output, f"planes/{stem}/output-{name}.u8", board,
                                                           ranges[name]) for name, board in actual.items()}}
                    # Preserve full H.264 output for optional portable NVDEC validation.
                    if target == "h264":
                        raw_path = output / "raw" / (stem + ".i420")
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_bytes(full_raw)
                        record.update(expected=str(raw_path.relative_to(output)), expected_sha256=digest(full_raw))
                    variants.append(record)
                entry.update(status="pass", costs=manifest["costs"], coded_shape=[coded_height, coded_width],
                             coded_luma_samples=manifest["costs"]["total_pictures"]*coded_width*coded_height,
                             prediction_sha256=digest(tail), variants=variants,
                             named_output_pixels_per_input=len(graph.outputs)*graph.width*graph.height,
                             synthesis_goals=[record["operation"] for record in program.synthesis_records])
            cases.append(entry)
            suffix = (f"{entry['costs']['computation_pictures']} compute + {entry['costs']['input_pictures']} input pictures"
                      if entry["status"] == "pass" else entry["reason"])
            print(f"{task.name:18s} {target:4s} {entry['status']}: {suffix}", flush=True)
    for path in source_paths:
        if sources[str(path.relative_to(ROOT))] != digest(path.read_bytes()):
            raise AssertionError("Source changed during comparison: " + str(path))
    if serializer and serializer["binary_sha256"] != digest(executable.read_bytes()):
        raise AssertionError("AV1 serializer changed during comparison")
    passed = [case for case in cases if case["status"] == "pass"]
    report = {"format": "codec-compute-workload-survey-1", "status": "pass", "timings": "not measured",
              "hardware_validation": "separate; this runner validates local software and available H.264 VA-API",
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": sources, "python": platform.python_version(),
              "av1_serializer": serializer,
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "task_target_pairs": len(cases), "supported_pairs": len(passed),
              "streams": sum(len(case["variants"]) for case in passed),
              "decoder_runs": sum(len(variant["decoders"]) for case in passed for variant in case["variants"]),
              "named_output_pixels": sum(case["named_output_pixels_per_input"]*len(case["variants"]) for case in passed),
              "semantics": "Identical traced graph on every target; explicit borders; independent task formulas",
              "cost_notes": "Pictures include inputs. Prediction bytes exclude input prefix; framing and geometry differ by codec.",
              "cases": cases}
    write_json(output / "validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "codec_compute" / "artifacts" / "workloads")
    parser.add_argument("--target", choices=("all", *TARGETS), default="all")
    parser.add_argument("--tasks", nargs="+", choices=[task.name for task in WORKLOADS])
    args = parser.parse_args()
    report = run(args.output, targets=TARGETS if args.target == "all" else (args.target,), tasks=args.tasks)
    print(json.dumps({key: report[key] for key in ("task_target_pairs", "supported_pairs", "streams",
                                                  "decoder_runs", "named_output_pixels")}, indent=2))
