#!/usr/bin/env python3
"""Portable temporal benchmark runner: Python 3.8+, stdlib, decoder_core only.

Run this file directly on a decode host; importing codec_compute is unnecessary.
Each batch is finite and includes process/pipeline startup. No latency claim is
made for resident streaming, GPU arithmetic, or a continuously running decoder.
"""

import __future__
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types

for parent in Path(__file__).resolve().parents:
    if (parent / "decoder_core" / "decode.py").is_file():
        sys.path.insert(0, str(parent))
        break
from decoder_core import decode as decoder


TASKS = ("positive-change", "absolute-change", "temporal-envelope", "temporal-average")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def checked(root, record):
    relative = record["path"]
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError("Artifact path escapes bundle")
    data = path.read_bytes()
    if digest(data) != record["sha256"] or len(data) != record["bytes"]:
        raise ValueError("Artifact hash/size mismatch: " + relative)
    return data


def oracle(task, a, b):
    """Byte arithmetic independent of the IR and decoder instruction selection."""
    if len(a) != len(b):
        raise ValueError("Input planes differ in size")
    if task == "positive-change":
        return {"output": bytes(max(x-y, 0) for x, y in zip(a, b))}
    if task == "absolute-change":
        values = bytes(abs(x-y) for x, y in zip(a, b))
        return {"output": values, "mask": bytes(int(v >= 24) for v in values)}
    if task == "temporal-envelope":
        return {"low": bytes(min(x, y) for x, y in zip(a, b)),
                "high": bytes(max(x, y) for x, y in zip(a, b)),
                "output": bytes(abs(x-y) for x, y in zip(a, b))}
    if task == "temporal-average":
        return {"output": bytes((x+y+1)//2 for x, y in zip(a, b))}
    raise ValueError("Unknown temporal task")


def geometry(manifest):
    height, width = manifest["ir"]["shape"]
    coded_height, coded_width = manifest["coded_shape"]
    guard, pictures = manifest["guard"], manifest["costs"]["total_pictures"]
    if any(type(v) is not int or v < 1 for v in (width, height, coded_width, coded_height, guard, pictures)):
        raise ValueError("Geometry and picture count must be positive integers")
    if coded_width != width + 2*guard or coded_height != height + 2*guard or coded_width % 2 or coded_height % 2:
        raise ValueError("Unsupported guarded I420 geometry")
    outputs = manifest["output_frames"]
    if not outputs or any(type(v) is not int or not 0 <= v < pictures for v in outputs.values()):
        raise ValueError("Invalid named output picture")
    return {"width": width, "height": height, "coded_width": coded_width, "coded_height": coded_height,
            "guard": guard, "pictures_per_job": pictures, "frame_bytes": coded_width*coded_height*3//2,
            "output_frames": outputs, "output_picture_indices": sorted(set(outputs.values()))}


def extract(frame, shape):
    stride, guard = shape["coded_width"], shape["guard"]
    return b"".join(frame[y*stride+guard:y*stride+guard+shape["width"]]
                    for y in range(guard, guard+shape["height"]))


def validate_guards(frames, shape):
    """Check zero luma outside the active image, plus neutral U and V."""
    stride, rows, guard = shape["coded_width"], shape["coded_height"], shape["guard"]
    luma = stride*rows
    spans = [(0, guard*stride), ((guard+shape["height"])*stride, luma)]
    spans += [(y*stride, y*stride+guard) for y in range(guard, guard+shape["height"])]
    spans += [(y*stride+guard+shape["width"], (y+1)*stride)
              for y in range(guard, guard+shape["height"])]
    for frame in frames:
        if len(frame) != shape["frame_bytes"] or frame[luma:] != bytes([128])*(luma//2):
            raise AssertionError("Invalid I420 size or nonneutral chroma")
        if any(frame[start:end] != bytes(end-start) for start, end in spans):
            raise AssertionError("Nonzero guard luma")
    return {"guard_bytes": len(frames)*(luma-shape["width"]*shape["height"]),
            "chroma_bytes": len(frames)*luma//2}


def counters(result):
    return {"decoder_sessions": 1, "decoded_pictures": result.decoded_frames,
            "copied_pictures": result.copied_frames,
            "decoded_i420_bytes": result.decoded_frames*result.frame_bytes,
            "copied_i420_bytes": result.copied_frames*result.frame_bytes,
            "returned_pictures": len(result.frames)*len(result.selected_frames),
            "returned_i420_bytes": sum(len(frame) for frames in result.frames for frame in frames)}


def writer_class(path):
    # Postpone type annotation evaluation so the identical Writer source also
    # runs on Python 3.8 (tuple[...] annotations otherwise require Python 3.9).
    # This changes no executable encoder logic and never edits the source file.
    module = types.ModuleType("_temporal_benchmark_writer")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(), str(path), "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), module.__dict__)
    return module.Writer


def load_bundle(root):
    bundle_bytes = (root / "bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle.get("format") != "codec-compute-temporal-bundle-1" or bundle.get("status") != "pass":
        raise ValueError("Expected a passing temporal benchmark bundle")
    if set(bundle["tasks"]) != set(TASKS):
        raise ValueError("Bundle must contain exactly the four temporal tasks")
    workloads = {}
    for task, item in bundle["tasks"].items():
        manifest = json.loads(checked(root, item["manifest"]))
        shape = geometry(manifest)
        tail = checked(root, item["tail"])
        if len(tail) != manifest["costs"]["prediction_bytes"]:
            raise AssertionError("Prediction byte count differs")
        jobs = []
        for index, job in enumerate(item["jobs"]):
            if job["index"] != index:
                raise ValueError("Job indices must be consecutive")
            stream = checked(root, job["stream"])
            prefix = checked(root, job["input_prefix"])
            if stream != prefix + tail:
                raise AssertionError("Stream differs from input prefix plus fixed prediction tail")
            inputs = {name: checked(root, record) for name, record in job["inputs"].items()}
            if set(inputs) != {"a", "b"} or any(len(value) != shape["width"]*shape["height"] for value in inputs.values()):
                raise ValueError("Expected two complete byte input planes")
            outputs = {name: checked(root, record) for name, record in job["outputs"].items()}
            if outputs != oracle(task, inputs["a"], inputs["b"]) or set(outputs) != set(shape["output_frames"]):
                raise AssertionError("Saved outputs differ from independent task formula")
            raw = checked(root, job["expected"])
            if len(raw) != shape["frame_bytes"]*shape["pictures_per_job"]:
                raise AssertionError("Full decoded reference size differs")
            frames = tuple(raw[i:i+shape["frame_bytes"]] for i in range(0, len(raw), shape["frame_bytes"]))
            validate_guards(frames, shape)
            for i, name in enumerate(("a", "b")):
                if extract(frames[i], shape) != inputs[name]:
                    raise AssertionError("Saved input picture differs from fixture")
            for name, frame in shape["output_frames"].items():
                if extract(frames[frame], shape) != outputs[name]:
                    raise AssertionError("Saved full reference differs from named output")
            boards = {name: [list(value[y*shape["width"]:(y+1)*shape["width"]])
                             for y in range(shape["height"])] for name, value in inputs.items()}
            jobs.append({"path": root / job["stream"]["path"], "prefix": prefix,
                         "frames": frames, "outputs": outputs, "boards": boards})
        if len(jobs) != bundle["job_count"]:
            raise ValueError("Job count differs from bundle")
        if len({job["input_prefix"]["sha256"] for job in item["jobs"]}) != len(jobs):
            raise AssertionError("Benchmark jobs must have distinct input pairs")
        workloads[task] = {"shape": shape, "tail": tail, "jobs": jobs, "record": item}
    return bundle_bytes, bundle, workloads


def run(bundle_path, output, *, backend="ffmpeg", repeats=5, batch_sizes=(1, 16), helper=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    if not batch_sizes or len(set(batch_sizes)) != len(batch_sizes) or any(type(n) is not int or n < 1 for n in batch_sizes):
        raise ValueError("batch sizes must be distinct positive integers")
    if helper is not None:
        if backend != "nvdec":
            raise ValueError("--helper applies only to NVDEC")
        os.environ["H264_NVDEC_DECODER"] = str(Path(helper).resolve())
    if not decoder.available(backend):
        raise RuntimeError("Decoder unavailable: " + backend)
    root, output = Path(bundle_path).resolve(), Path(output).resolve()
    bundle_bytes, bundle, workloads = load_bundle(root)
    if max(batch_sizes) > bundle["job_count"]:
        raise ValueError("Batch size exceeds independent prepared jobs")
    writer_path = Path(decoder.__file__).with_name("bitstream.py")
    source_files = {"runner": Path(__file__), "decoder": Path(decoder.__file__), "writer": writer_path}
    source_hashes = {name: digest(path.read_bytes()) for name, path in source_files.items()}
    if source_hashes != bundle["runtime_source_sha256"]:
        raise AssertionError("Runner/decoder/writer source differs from prepared bundle")
    Writer = writer_class(writer_path)
    for workload in workloads.values():
        params = dict(workload["record"]["writer"])
        params["reference_slots"] = {int(k): v for k, v in params["reference_slots"].items()}
        workload["writer"] = Writer(**params)
    report = {"format": "codec-compute-temporal-benchmark-1", "status": "running", "backend": backend,
              "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              "bundle_sha256": digest(bundle_bytes), "repository_commit": bundle["repository_commit"],
              "runtime_source_sha256": source_hashes, "repeats": repeats, "warmups_per_mode": 1,
              "batch_sizes": list(batch_sizes), "tasks": {},
              "protocol": {"input-serialization": "Two I_PCM input pictures per job, Writer validation and exact prefix comparison; inputs already loaded as integer boards; no disk I/O, compile or decode",
                           "preencoded-all": "Batch file assembly/I/O, one process and decoder startup, parsing, all picture decode/readback, diagnostics, exact I420 comparison and named-output extraction/comparison",
                           "preencoded-outputs": "Same decode scope, requesting only deduplicated named-output pictures; NVDEC copies selected surfaces, FFmpeg copies every picture then selects in Python",
                           "host-to-outputs": "Measured together: input serialization and exact prefix comparison, stream assembly/write, then preencoded-outputs scope; fixed prediction tail is cached",
                           "excluded": "Compilation, fixture/reference generation and loading, source/hash audit, initial full guard/chroma validation, network, capture writing",
                           "ordering": "Rotate task/batch/mode order each repetition, after one warmup each",
                           "caveat": "Finite synchronous batches to EOS, each with startup. Per-job values are amortized throughput, not single-job latency. No resident stream, GPU baseline or concurrent GPU workload is measured.",
                           "writer_compatibility": "Identical Writer source compiled with postponed annotations for Python 3.8; executable encoder logic unchanged and every generated prefix checked byte for byte"}}
    if backend == "nvdec":
        report["decoder_helper_sha256"] = digest(decoder._nvdec_helper().read_bytes())
    else:
        report["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], universal_newlines=True).splitlines()[0]

    def save():
        write_json(output, report)

    def serialize(workload, jobs):
        writer = workload["writer"]
        prefixes = []
        for job in jobs:
            prefix = b"".join(writer.input_luma(job["boards"][name], index) for index, name in enumerate(("a", "b")))
            if prefix != job["prefix"]:
                raise AssertionError("Timed serialization differs from prepared input prefix")
            prefixes.append(prefix)
        return prefixes

    def decode(workload, jobs, paths, selection):
        shape = workload["shape"]
        result = decoder.decode_batch(paths, backend, width=shape["coded_width"], height=shape["coded_height"],
                                      pictures_per_job=shape["pictures_per_job"], selected_frames=selection)
        if len(result.frames) != len(jobs):
            raise AssertionError("Decoded job count differs")
        positions = {frame: i for i, frame in enumerate(result.selected_frames)}
        for frames, job in zip(result.frames, jobs):
            if frames != tuple(job["frames"][i] for i in result.selected_frames):
                raise AssertionError("Actual decoded I420 differs from full saved reference")
            for name, frame in shape["output_frames"].items():
                if extract(frames[positions[frame]], shape) != job["outputs"][name]:
                    raise AssertionError("Actual named output differs from independent formula")
        return result

    # Full independent validation is separate from the timed trials.
    for task, workload in workloads.items():
        jobs, shape = workload["jobs"], workload["shape"]
        result = decode(workload, jobs, [job["path"] for job in jobs], None)
        guard_counts = {"guard_bytes": 0, "chroma_bytes": 0}
        for frames in result.frames:
            for key, count in validate_guards(frames, shape).items():
                guard_counts[key] += count
        report["tasks"][task] = {"geometry": shape, "costs": workload["record"]["costs"],
                                 "validation": {"status": "pass", **counters(result), **guard_counts,
                                                "named_outputs": len(jobs)*len(shape["output_frames"])}, "batches": {}}
        print("PASS %s: full I420, inputs, named outputs, guards and chroma" % task, flush=True)
    save()

    def execute(task, count, mode):
        workload = workloads[task]
        jobs = workload["jobs"][:count]
        if mode == "input-serialization":
            prefixes = serialize(workload, jobs)
            return {"encoded_input_pictures": count*2, "encoded_input_bytes": sum(map(len, prefixes)), "decoder_sessions": 0}
        selection = None if mode == "preencoded-all" else workload["shape"]["output_picture_indices"]
        if mode == "host-to-outputs":
            prefixes = serialize(workload, jobs)
            with tempfile.TemporaryDirectory(prefix="temporal-inputs-") as directory:
                paths = [Path(directory) / ("%03d.h264" % i) for i in range(count)]
                for path, prefix in zip(paths, prefixes):
                    path.write_bytes(prefix + workload["tail"])
                result = decode(workload, jobs, paths, selection)
            return {**counters(result), "encoded_input_pictures": count*2, "encoded_input_bytes": sum(map(len, prefixes))}
        return counters(decode(workload, jobs, [job["path"] for job in jobs], selection))

    runs = [(task, count, mode) for task in workloads for count in batch_sizes
            for mode in ("input-serialization", "preencoded-all", "preencoded-outputs", "host-to-outputs")]
    samples = {key: [] for key in runs}
    for iteration in range(repeats+1):
        offset = iteration % len(runs)
        for task, count, mode in runs[offset:] + runs[:offset]:
            start = time.perf_counter()
            actual = execute(task, count, mode)
            elapsed = (time.perf_counter()-start)*1000
            if iteration:
                values = samples[task, count, mode]
                values.append(elapsed)
                median = statistics.median(values)
                report["tasks"][task]["batches"].setdefault(str(count), {})[mode] = {
                    "status": "pass", **actual, "jobs": count, "samples_ms": list(values),
                    "median_ms": median, "min_ms": min(values), "max_ms": max(values),
                    "amortized_ms_per_job": median/count, "jobs_per_second": 1000*count/median}
                save()
            print("PASS %s %d %s %s: %.3f ms" %
                  (task, count, mode, "warmup" if not iteration else "run %d" % iteration, elapsed), flush=True)
    if (root / "bundle.json").read_bytes() != bundle_bytes:
        raise AssertionError("Bundle changed during benchmark")
    if {name: digest(path.read_bytes()) for name, path in source_files.items()} != source_hashes:
        raise AssertionError("Runtime source changed during benchmark")
    report["status"] = "pass"
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("ffmpeg", "nvdec"), default="ffmpeg")
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    args = parser.parse_args()
    run(args.bundle, args.output, backend=args.backend, repeats=args.repeats,
        batch_sizes=args.batch_sizes, helper=args.helper)
