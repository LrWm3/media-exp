"""Export exact static-field resampling comparisons and portable timing inputs."""

import argparse
import json
from pathlib import Path
import subprocess

import codec_compute as media
from codec_compute.targets.h264 import _legacy
from decoder_core.decode import available, decode_batch
from ..temporal_benchmark import digest, geometry, validate_guards, write_json
from .kernels import fixture, quality_errors, quality_fixture, quality_kernel, reference, tasks, warp


ROOT = Path(__file__).resolve().parents[3]


def prepare(output, *, size=128, jobs=8):
    if type(jobs) is not int or jobs < 2:
        raise ValueError("At least two changing fixtures are required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for path in (ROOT / "codec_compute").rglob("*.py")
                     if not {"tests", "artifacts", "assets"}.intersection(path.parts))
    sources += sorted((ROOT / "05-h264-compiler" / "h264").glob("*.py"))
    sources += [ROOT / "decoder_core" / name for name in ("bitstream.py", "decode.py")]
    source_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    decoders = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
    if "ffmpeg" not in decoders:
        raise RuntimeError("FFmpeg is required for the frozen full reference")

    def artifact(relative, data):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": relative, "sha256": digest(data), "bytes": len(data)}

    def json_artifact(relative, value):
        return artifact(relative, (json.dumps(value, indent=2)+"\n").encode())

    def plane(relative, board, bounds=(0, 255)):
        return {**artifact(relative, bytes(v for row in board for v in row)),
                "width": len(board[0]), "height": len(board), "range": list(bounds)}

    cases, comparisons, suite_cases = {}, [], []
    for task in tasks(size):
        fields = json_artifact(task.name + "/fields.json", {"block_size": 16, "border": "zero", "fields": task.fields})
        images = [fixture(task, index, size) for index in range(jobs)]
        expected = [reference(task, inputs) for inputs in images]
        input_records = [{name: plane("%s/inputs/%03d-%s.u8" % (task.name, index, name), board)
                          for name, board in inputs.items()} for index, inputs in enumerate(images)]
        variants = (("fused", True),) if task.name.endswith("control") else (("baseline", False), ("fused", True))
        for variant, enabled in variants:
            key = task.name + "/" + variant
            program = media.compile(task.kernel, target="h264", width=size, height=size,
                                    input_ranges=task.input_ranges, h264_fuse_shifts=enabled)
            manifest, tail = program.manifest(), program.tail
            shape = geometry(manifest)
            manifest_record = json_artifact(key + "/manifest.json", manifest)
            tail_record = artifact(key + "/prediction.h264", tail)
            ranges = {name: (node.lo, node.hi) for name, node in program.graph.outputs}
            paths, records = [], []
            for index, inputs in enumerate(images):
                if program.reference(inputs) != expected[index]:
                    raise AssertionError(key + ": semantic IR differs from independent task formula")
                stream = program.encode(inputs)
                if program.tail != tail or not stream.endswith(tail):
                    raise AssertionError("Computation tail changed with input pixels")
                stem = "%s/%03d" % (key, index)
                record = {"index": index, "inputs": input_records[index],
                          "input_prefix": artifact(stem + ".prefix", stream[:-len(tail)] if tail else stream),
                          "stream": artifact(stem + ".h264", stream), "decoders": []}
                records.append(record)
                paths.append(output / record["stream"]["path"])
            frozen = None
            for decoder in decoders:
                result = decode_batch(paths, decoder, width=shape["coded_width"], height=shape["coded_height"],
                                      pictures_per_job=shape["pictures_per_job"])
                if frozen is not None and result.frames != frozen:
                    raise AssertionError(key + ": decoders differ in full I420")
                frozen = result.frames
                for index, (record, frames) in enumerate(zip(records, result.frames)):
                    raw = b"".join(frames)
                    actual = _legacy().backend.read(raw, program.native)
                    if actual != expected[index]:
                        raise AssertionError(key + ": decoded output differs from independent formula")
                    validate_guards(frames, shape)
                    record["decoders"].append({"backend": decoder, "status": "pass", "i420_sha256": digest(raw),
                                               "decoded_pictures": len(frames), "decoded_bytes": len(raw)})
                    if decoder == "ffmpeg":
                        stem = "%s/%03d" % (key, index)
                        record["expected"] = artifact(stem + ".i420", raw)
                        record["outputs"] = {name: plane(stem + "-" + name + ".u8", board, ranges[name])
                                             for name, board in actual.items()}
            for record in records:
                suite_cases.append({"name": key + "/%03d" % record["index"],
                                    "stream": record["stream"]["path"], "stream_sha256": record["stream"]["sha256"],
                                    "manifest": manifest_record["path"], "manifest_sha256": manifest_record["sha256"],
                                    "expected": record["expected"]["path"], "expected_sha256": record["expected"]["sha256"],
                                    "pictures": shape["pictures_per_job"], "width": size, "height": size,
                                    "coded_width": shape["coded_width"], "coded_height": shape["coded_height"],
                                    "frame_bytes": shape["frame_bytes"],
                                    "named_outputs": {name: {**info, "frame": shape["output_frames"][name],
                                                              "x": shape["guard"], "y": shape["guard"]}
                                                      for name, info in record["outputs"].items()}})
            writer = program.native.writer
            cases[key] = {"task": task.name, "variant": variant, "title": task.title, "description": task.description,
                          "fields": fields, "input_names": list(task.input_names), "costs": manifest["costs"],
                          "geometry": shape, "manifest": manifest_record, "tail": tail_record, "jobs": records,
                          "writer": {"width": size, "height": size, "fps": writer.fps, "motion_bits": writer.motion_bits,
                                     "reference_slots": dict(writer.reference_slots), "slice_mode": writer.slice_mode}}
            print("PASS %s: %d compute + %d input pictures; %d prediction bytes" %
                  (key, manifest["costs"]["computation_pictures"], manifest["costs"]["input_pictures"], len(tail)), flush=True)
        if not task.name.endswith("control"):
            before, after = (cases[task.name + "/" + name] for name in ("baseline", "fused"))
            for key in ("computation_pictures", "prediction_bytes"):
                if after["costs"][key] > before["costs"][key]:
                    raise AssertionError("Fusion regressed " + key)
            for a, b in zip(before["jobs"], after["jobs"]):
                if {n: p["sha256"] for n, p in a["outputs"].items()} != {n: p["sha256"] for n, p in b["outputs"].items()}:
                    raise AssertionError("Fusion changed a named output")
            comparisons.append({"task": task.name, "baseline": before["costs"], "fused": after["costs"],
                                "outputs_equal": True, "coded_shape_equal": True})

    # A separate nonlinear known-motion witness evaluates usefulness, not only
    # exact interpolation arithmetic. Main block fields above stay independent.
    images, midpoint = quality_fixture(size)
    program = media.compile(quality_kernel, target="h264", width=size, height=size,
                            input_ranges={"a": (0, 255), "b": (0, 255)})
    manifest, stream = program.manifest(), program.encode(images)
    shape = geometry(manifest)
    quality_stream = artifact("quality/known-motion.h264", stream)
    quality_manifest = json_artifact("quality/manifest.json", manifest)
    quality_raw, quality_outputs, quality_decoders = None, None, []
    count = (size//16)**2
    first = warp(images["a"], ((10, 0),)*count, quarter=True)
    second = warp(images["b"], ((-10, 0),)*count, quarter=True)
    independent = {"compensated": [[(x+y+1)//2 for x, y in zip(a, b)] for a, b in zip(first, second)],
                   "naive": [[(x+y+1)//2 for x, y in zip(a, b)] for a, b in zip(images["a"], images["b"])]}
    if program.reference(images) != independent:
        raise AssertionError("Quality IR differs from normative interpolation reference")
    for decoder in decoders:
        result = decode_batch([output / quality_stream["path"]], decoder, width=shape["coded_width"],
                              height=shape["coded_height"], pictures_per_job=shape["pictures_per_job"])
        raw = b"".join(result.frames[0])
        actual = _legacy().backend.read(raw, program.native)
        if actual != independent or (quality_raw is not None and quality_raw != raw):
            raise AssertionError("Quality actual decode differs from exact reference")
        quality_raw, quality_outputs = raw, actual
        quality_decoders.append({"backend": decoder, "status": "pass", "i420_sha256": digest(raw)})
    quality = {"description": "Analytic sinusoidal texture translated by +/-2.5 pixels; supplied uniform half-pel compensation",
               "stream": quality_stream, "manifest": quality_manifest, "costs": manifest["costs"],
               "expected": artifact("quality/known-motion.i420", quality_raw), "decoders": quality_decoders,
               "inputs": {name: plane("quality/input-" + name + ".u8", board) for name, board in images.items()},
               "outputs": {name: plane("quality/" + name + ".u8", board) for name, board in quality_outputs.items()},
               "midpoint": plane("quality/analytic-midpoint.u8", midpoint), **quality_errors(quality_outputs, midpoint)}
    if quality["metrics"]["compensated"]["mae"] >= quality["metrics"]["naive"]["mae"]:
        raise AssertionError("Known-motion compensation did not improve this controlled witness")
    suite_cases.append({"name": "quality", "stream": quality_stream["path"], "stream_sha256": quality_stream["sha256"],
                        "manifest": quality_manifest["path"], "manifest_sha256": quality_manifest["sha256"],
                        "expected": quality["expected"]["path"], "expected_sha256": quality["expected"]["sha256"],
                        "pictures": shape["pictures_per_job"], "width": size, "height": size,
                        "coded_width": shape["coded_width"], "coded_height": shape["coded_height"],
                        "frame_bytes": shape["frame_bytes"],
                        "named_outputs": {name: {**info, "frame": shape["output_frames"][name], "x": shape["guard"], "y": shape["guard"]}
                                          for name, info in quality["outputs"].items()}})
    # Probe uploader limits explicitly with arbitrary byte patterns. This is
    # input serialization only; it makes no claim of large-image execution.
    probe = media.compile(lambda image: image, target="h264", width=256, height=256, input_ranges={"image": (0, 255)})
    upload_probes = []
    for label, board in (("zero", [[0]*256 for _ in range(256)]),
                         ("byte-ramp", [[(17*x+31*y) % 256 for x in range(256)] for y in range(256)])):
        try:
            prefix = probe.encode({"image": board})
        except ValueError as error:
            upload_probes.append({"shape": [256, 256], "fixture": label, "status": "rejected", "reason": str(error)})
        else:
            upload_probes.append({"shape": [256, 256], "fixture": label, "status": "accepted", "input_bytes": len(prefix)})
    if source_hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}:
        raise AssertionError("Source changed during export")
    runtime = {"runner": "codec_compute/workloads/resampling/benchmark.py", "common": "codec_compute/workloads/temporal_benchmark.py",
               "decoder": "decoder_core/decode.py", "writer": "decoder_core/bitstream.py"}
    report = {"format": "codec-compute-resampling-1", "status": "pass", "target": "h264", "size": size, "job_count": jobs,
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": source_hashes, "runtime_source_sha256": {key: source_hashes[path] for key, path in runtime.items()},
              "field_contract": "Externally supplied compile-time constants; identical prediction tail for every fixture; no motion estimation",
              "validation": "Semantic IR plus independent task formulas/legacy normative interpolation; full actual I420, all named outputs, zero guards and neutral chroma",
              "streams": len(suite_cases), "pictures": sum(case["pictures"] for case in suite_cases),
              "local_decoder_runs": len(suite_cases)*len(decoders), "decoders": decoders,
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "cases": cases, "comparisons": comparisons, "quality": quality, "upload_probes": upload_probes}
    write_json(output / "bundle.json", report)
    (output / "comparison.json").write_bytes((output / "bundle.json").read_bytes())
    write_json(output / "suite.json", {"format": "h264-ir-suite-0.1", "selection": "Every baseline/fused stream, passthrough control, and known-motion witness",
                                      "comparison_sha256": digest((output / "comparison.json").read_bytes()),
                                      "exporter_sha256": digest(Path(__file__).read_bytes()), "case_count": len(suite_cases), "cases": suite_cases})
    print("Quality MAE: " + json.dumps(quality["metrics"]), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "codec_compute/artifacts/resampling")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    prepare(args.output, size=args.size, jobs=args.jobs)
