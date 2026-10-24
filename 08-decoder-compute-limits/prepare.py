#!/usr/bin/env python3
"""Prepare portable literal-Boolean atlas and dependent-throughput streams.

The bundle contains frozen streams, full I420 reference pictures, independent
truth-table outputs, literal inputs, manifests, and source/model hashes. Its
consumer needs decoder_core only; no graph evaluator or atlas search is needed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

from atlas import build_atlas
from program import Program, RECURRENCE_NOTE
from decoder_core.decode import available, decode


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plane(board):
    return bytes(value for row in board for value in row)


def sample_inputs(width, height, seed=80008):
    """Every truth pair touches borders and interior; other lanes are random."""
    rng = random.Random(seed + width * 1000 + height)
    result = {name: [[rng.randrange(2) for _ in range(width)] for _ in range(height)]
              for name in ("a", "b")}
    for y in range(height):
        for x in range(width):
            if y in (0, height - 1) or x in (0, width - 1) or y == height // 2:
                index = (x + y) % 4
                result["a"][y][x], result["b"][y][x] = index // 2, index % 2
    return result


def save_bytes(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(relative), "sha256": digest(data), "bytes": len(data)}


def _validate_input_frames(program, raw, inputs):
    width, height = program.writer.width, program.writer.height
    stride = width * height * 3 // 2
    for frame, name in enumerate(("a", "b")):
        actual = b"".join(raw[frame * stride + (y + 16) * width + 16:
                              frame * stride + (y + 16) * width + 16 + program.width]
                          for y in range(program.height))
        if actual != plane(inputs[name]):
            raise AssertionError("Decoded literal input differs: " + name)


def prepare(output, decoders=("ffmpeg", "openh264", "vaapi"), *, throughput=True,
            sizes=(16, 32, 64), rounds=128):
    output = Path(output).resolve()
    if not available("ffmpeg"):
        raise RuntimeError("FFmpeg is required to prepare complete I420 references")
    if (type(rounds) is not int or rounds < 1 or not sizes
            or any(type(size) is not int or size < 16 or size % 16 for size in sizes)):
        raise ValueError("Throughput sizes must be multiples of 16 and rounds positive")
    requested = tuple(dict.fromkeys(("ffmpeg",) + tuple(decoders)))
    ready = tuple(name for name in requested if available(name))
    atlas = build_atlas()
    write_json(output / "atlas.json", atlas)
    proofs = save_bytes(output, Path("PROOFS.md"), (ROOT / "PROOFS.md").read_bytes())
    model = {"format": "h264-compute-limits-model-0.1", "sample_domain": [0, 1],
             "bit_depth": 8, "truth_order": ["00", "01", "10", "11"],
             "prediction_sources": [1, 2], "weight_range": [-128, 127],
             "effective_offset_range": [-128, 127], "denominator_range": [0, 7],
             "recurrence": "a[r+1] = truth[2*a[r] + original_b]",
             "recurrence_note": RECURRENCE_NOTE,
             "input_encoding": "two literal I_PCM luma pictures; chroma 128; guards 0",
             "compute_encoding": "weighted prediction; zero residual; deblocking disabled",
             "atlas_sha256": digest((output / "atlas.json").read_bytes())}
    write_json(output / "model.json", model)
    sources = sorted([*ROOT.glob("*.py"), ROOT / "PROOFS.md", ROOT.parent / "decoder_core" / "bitstream.py",
                      ROOT.parent / "decoder_core" / "decode.py"])
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                           cwd=ROOT, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                                      cwd=ROOT, text=True).strip()),
                  "source_sha256": {str(path.relative_to(ROOT.parent)): digest(path.read_bytes())
                                    for path in sources}}
    write_json(output / "provenance.json", provenance)
    report = {"format": "h264-compute-limits-bundle-0.1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "atlas": "atlas.json", "atlas_sha256": model["atlas_sha256"],
              "proofs": proofs["path"], "proofs_sha256": proofs["sha256"],
              "model": "model.json", "model_sha256": digest((output / "model.json").read_bytes()),
              "provenance": "provenance.json",
              "provenance_sha256": digest((output / "provenance.json").read_bytes()),
              "input_seed": 80008, "input_encoding": "literal Boolean luma 0/1",
              "decoders": {name: "available" if name in ready else "unavailable" for name in requested},
              **provenance, "cases": []}
    jobs = [(case, mode, "atlas", 32, 1) for case in atlas["functions"]
            for mode in ("scalar", "packed")]
    if throughput:
        by_truth = {tuple(case["truth"]): case for case in atlas["functions"]}
        kernels = [(by_truth[(1, 1, 1, 0)], "scalar"),
                   (by_truth[(0, 1, 1, 0)], "scalar"),
                   (by_truth[(0, 1, 1, 0)], "packed")]
        jobs.extend((case, mode, "throughput", size, rounds)
                    for case, mode in kernels for size in dict.fromkeys(sizes))
        if 32 in sizes and rounds != 32:
            jobs.extend((case, mode, "throughput", 32, 32) for case, mode in kernels)
    for case, mode, purpose, size, count in jobs:
        identifier = "%s-%02d-%s-%dx%d-r%d" % (purpose, case["id"], mode, size, size, count)
        directory = Path("cases") / identifier
        started = time.perf_counter()
        program = Program(case, width=size, height=size, packed=mode == "packed", rounds=count)
        lowering_ms = (time.perf_counter() - started) * 1000
        inputs = sample_inputs(size, size)
        expected = program.reference(inputs)
        started = time.perf_counter()
        tail = program.tail
        tail_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        prefix = program.encode_inputs(inputs)
        prefix_ms = (time.perf_counter() - started) * 1000
        stream = prefix + tail
        if program.encode(inputs) != stream:
            raise AssertionError("Program did not reuse its immutable prediction tail")
        stored_inputs = {name: save_bytes(output, directory / (name + ".u8"), plane(board))
                         for name, board in inputs.items()}
        saved_stream = save_bytes(output, directory / "program.h264", stream)
        saved_tail = save_bytes(output, directory / "prediction.h264", tail)
        manifest_path = directory / "program.json"
        write_json(output / manifest_path, program.manifest())
        raw = decode(output / saved_stream["path"], "ffmpeg")
        if program.read(raw) != expected:
            raise AssertionError(identifier + ": FFmpeg differs from independent truth-table oracle")
        _validate_input_frames(program, raw, inputs)
        frame_bytes = program.writer.width * program.writer.height * 3 // 2
        checks = {}
        for backend in ready:
            decoded = raw if backend == "ffmpeg" else decode(output / saved_stream["path"], backend)
            if decoded != raw:
                raise AssertionError(identifier + ": full I420 bytes differ for " + backend)
            if program.read(decoded) != expected:
                raise AssertionError(identifier + ": truth-table output differs for " + backend)
            checks[backend] = {"status": "pass", "pictures": program.pictures,
                               "i420_bytes": len(raw), "output_pixels": size * size,
                               "full_planes_compared": True}
        saved_expected = save_bytes(output, directory / "expected.i420", raw)
        saved_output = save_bytes(output, directory / "output.u8", plane(expected["output"]))
        record = {"id": identifier, "function": case["name"], "mask": case["id"],
                  "mode": mode, "purpose": purpose, "width": size, "height": size,
                  "coded_width": program.writer.width, "coded_height": program.writer.height,
                  "pictures": program.pictures, "rounds": count, "evaluations": count * size * size,
                  "output_frames": program.output_frames, "output_regions": program.output_regions,
                  "manifest": str(manifest_path), "manifest_sha256": digest((output / manifest_path).read_bytes()),
                  "stream": saved_stream["path"], "stream_sha256": saved_stream["sha256"],
                  "stream_bytes": saved_stream["bytes"], "input_bytes": len(prefix),
                  "expected": saved_expected["path"], "expected_sha256": saved_expected["sha256"],
                  "expected_bytes": saved_expected["bytes"],
                  "expected_output": saved_output["path"], "expected_output_sha256": saved_output["sha256"],
                  "expected_output_bytes": saved_output["bytes"], "frame_bytes": frame_bytes,
                  "inputs": stored_inputs, "prediction": saved_tail, "costs": program.costs,
                  "single_evaluation_minima": program.single_evaluation_minima,
                  "cost_scope": "executed witness counts across all rounds",
                  "recurrence_note": RECURRENCE_NOTE,
                  "lowering_ms": lowering_ms, "first_tail_ms": tail_ms, "input_packing_ms": prefix_ms,
                  "decoders": checks}
        report["cases"].append(record)
        print("PASS %s: %d pictures, %dx%d coded, %d prediction bytes, %s" %
              (identifier, program.pictures, program.writer.width, program.writer.height,
               len(tail), ", ".join(ready)), flush=True)
    write_json(output / "bundle.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--decoders", nargs="+", choices=("ffmpeg", "openh264", "vaapi"),
                        default=["ffmpeg", "openh264", "vaapi"])
    parser.add_argument("--atlas-only", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--rounds", type=int, default=128)
    args = parser.parse_args()
    result = prepare(args.output, args.decoders, throughput=not args.atlas_only,
                     sizes=tuple(args.sizes), rounds=args.rounds)
    print(json.dumps({"cases": len(result["cases"]), "decoders": result["decoders"]}))


if __name__ == "__main__":
    main()
