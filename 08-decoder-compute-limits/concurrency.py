#!/usr/bin/env python3
"""Measure fixed CUDA work plus eight exact denoising results, using Python 3.8.

Only stdlib and decoder_core are imported. The immutable prepared bundle and
weight JSON supply inputs, outputs and model parameters. CUDA is an execution
proxy for shader workloads; this does not exercise a graphics API.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import select
import statistics
import subprocess
import sys
import tempfile
import time

for parent in Path(__file__).resolve().parents:
    if (parent / "decoder_core" / "decode.py").is_file():
        sys.path.insert(0, str(parent))
        break
from decoder_core.decode import _nvdec_helper, decode_batch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(root, relative, checksum):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Bundle path escapes directory")
    data = path.read_bytes()
    if digest(data) != checksum:
        raise ValueError("Bundle checksum mismatch")
    return data


def model_parameters(model):
    if (model.get("format") != "binary-denoiser-v1" or model.get("padding") != "zero"
            or model.get("activation") != "sum >= integer threshold"):
        raise ValueError("Unsupported frozen model semantics")
    kernels, thresholds, mix = (model[key] for key in ("kernels", "thresholds", "mix_weights"))
    if (len(kernels) != 8 or any(len(kernel) != 9 for kernel in kernels)
            or len(thresholds) != 8 or len(mix) != 8):
        raise ValueError("Expected eight 3x3 channels and one eight-channel mix")
    binary = [value for kernel in kernels for value in kernel] + list(mix)
    if any(type(value) is not int or value not in (-1, 1) for value in binary):
        raise ValueError("Expected binary integer weights")
    if any(type(value) is not int or not -1024 <= value <= 1024
           for value in list(thresholds) + [model["mix_threshold"]]):
        raise ValueError("Expected bounded integer thresholds")
    return [value for kernel in kernels for value in kernel] + list(thresholds) + list(mix) + [model["mix_threshold"]]


def load_bundle(bundle_path, weights_path):
    root = Path(bundle_path).resolve()
    bundle_bytes = (root / "bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    weights_bytes = Path(weights_path).read_bytes()
    if (bundle.get("format") != "h264-packed-denoiser-bundle-0.1"
            or bundle["job_count"] != 8 or len(bundle["inputs"]) != 8):
        raise ValueError("Expected the eight-image packed denoiser bundle")
    if digest(weights_bytes) != bundle["model_sha256"]:
        raise ValueError("Model differs from prepared bundle")
    parameters = model_parameters(json.loads(weights_bytes))
    variant = bundle["variants"]["packed-8"]
    manifest = json.loads(checked_file(root, variant["manifest"], variant["manifest_sha256"]))
    if ((manifest["width"], manifest["height"], manifest["coded_width"], manifest["coded_height"],
         manifest["pictures"], manifest["guard"]) != (32, 32, 208, 112, 17, 16)):
        raise ValueError("Unexpected packed-8 geometry")
    region = manifest["output_regions"]["output"]
    if (region["frame"], region["x"], region["y"], region["width"], region["height"]) != (16, 0, 0, 32, 32):
        raise ValueError("Unexpected final output region")
    inputs, oracles, paths = [], [], []
    if len(variant["jobs"]) != 8:
        raise ValueError("Expected eight independent decoder streams")
    for index, (item, job) in enumerate(zip(bundle["inputs"], variant["jobs"])):
        if item["index"] != index or job["index"] != index or job["input_sha256"] != item["sha256"]:
            raise ValueError("Input ordering differs between implementations")
        data = checked_file(root, item["path"], item["sha256"])
        oracle = item["named_outputs"]["output"]
        expected = checked_file(root, oracle["path"], oracle["sha256"])
        if len(data) != 1024 or len(expected) != 1024 or any(value > 1 for value in data + expected):
            raise ValueError("Expected 32x32 binary input and output")
        checked_file(root, job["stream"], job["stream_sha256"])
        inputs.append(data); oracles.append(expected); paths.append(root / job["stream"])
    return {"input": b"".join(inputs), "expected": b"".join(oracles), "oracles": oracles,
            "paths": paths, "parameters": parameters, "bundle_sha256": digest(bundle_bytes),
            "weights_sha256": digest(weights_bytes), "input_sha256": [digest(value) for value in inputs],
            "output_sha256": [digest(value) for value in oracles]}


def extract_final(frame):
    if len(frame) != 208 * 112 * 3 // 2:
        raise AssertionError("Final I420 byte count differs")
    return b"".join(frame[(y + 16) * 208 + 16:(y + 16) * 208 + 48] for y in range(32))


class CUDAService:
    def __init__(self, helper, ptx, bundle, directory, elements, cycles):
        self.directory = Path(directory)
        (self.directory / "input.u8").write_bytes(bundle["input"])
        (self.directory / "weights.txt").write_text(" ".join(map(str, bundle["parameters"])) + "\n")
        self.output = self.directory / "output.u8"
        self.errors = (self.directory / "stderr.txt").open("w+")
        self.process = subprocess.Popen(
            [str(Path(helper).resolve()), str(Path(ptx).resolve()), str(self.directory / "input.u8"),
             str(self.directory / "weights.txt"), str(self.output), str(elements), str(cycles), "protocol-v1"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors, bufsize=0)
        self.buffer = b""
        try:
            self.read("ready")
        except Exception:
            self.close()
            raise

    def read(self, expected, timeout=180):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("CUDA service response timed out")
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 1))
            if ready:
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data:
                    self.errors.flush(); self.errors.seek(0)
                    raise RuntimeError("CUDA service stopped: " + self.errors.read().strip())
                self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        value = json.loads(line)
        if value.get("event") != expected:
            raise RuntimeError("Unexpected CUDA service event: " + str(value.get("event")))
        return value

    def command(self, command, event):
        self.process.stdin.write((command + "\n").encode())
        self.process.stdin.flush()
        return self.read(event)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b"QUIT\n"); self.process.stdin.flush()
                self.process.wait(timeout=10)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill(); self.process.wait()
        self.process.stdin.close(); self.process.stdout.close(); self.errors.close()


def overlap_statistics(stress, inference):
    """Host submission/completion intervals: evidence of overlap, not a profiler."""
    start, end = inference["start_ns"], inference["end_ns"]
    fully_inside = [unit for unit in stress["units"] if unit[0] >= start and unit[1] <= end]
    intersections = [max(0, min(end, unit[1]) - max(start, unit[0])) for unit in stress["units"]]
    duration = end - start
    return {"inference_interval_ns": [start, end],
            "work_interval_ns": [stress["start_ns"], stress["end_ns"]],
            "intersecting_units": sum(value > 0 for value in intersections),
            "fully_contained_units": len(fully_inside),
            "host_interval_overlap_ms": sum(intersections) / 1e6,
            "inference_interval_covered_fraction": sum(intersections) / duration if duration else 0,
            "contained_units_per_second": len(fully_inside) * 1e9 / duration if fully_inside and duration else None,
            "meaning": "overlapping host submission/completion intervals; not proof of simultaneous GPU instructions"}


def benchmark(args):
    if args.repeats not in (3, 5) or not 0.2 <= args.seconds <= 20:
        raise ValueError("Use 3 or 5 repeats and 0.2..20 target seconds")
    if not 256 <= args.elements <= 64 * 1024 * 1024 or not 1 <= args.cycles <= 4096:
        raise ValueError("CUDA workload size outside bounded limits")
    bundle = load_bundle(args.bundle, args.weights)
    if args.helper:
        os.environ["H264_NVDEC_DECODER"] = str(Path(args.helper).resolve())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"format": "h264-cuda-concurrency-0.1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "bundle_sha256": bundle["bundle_sha256"], "model_sha256": bundle["weights_sha256"],
              "input_sha256": bundle["input_sha256"], "output_sha256": bundle["output_sha256"],
              "cuda_helper_sha256": digest(Path(args.cuda_helper).read_bytes()),
              "ptx_sha256": digest(Path(args.ptx).read_bytes()),
              "nvdec_helper_sha256": digest(_nvdec_helper().read_bytes()),
              "source_sha256": {name: digest((Path(__file__).parent / name).read_bytes())
                                for name in ("concurrency.py", "gpu_workload.cu", "build_gpu_workload.sh")},
              "method": {"repeats": args.repeats, "target_gpu_seconds": args.seconds,
                         "batch_images": 8, "input_dimensions": [32, 32], "final_picture_only": 16,
                         "decoded_pictures_per_batch": 136, "copied_pictures_per_batch": 8,
                         "inference_validation": "every returned binary pixel against the frozen model bundle oracle, every run",
                         "cuda_initialization": "persistent shared context, independent nonblocking streams; allocations, pinned host buffers, PTX loading, model upload and process startup excluded",
                         "cuda_timed": "command IPC, 8192-byte input H2D, fused exact inference kernel, 8192-byte D2H, output file write/read, oracle check; stage CUDA events also reported",
                         "nvdec_timed": "decode_batch stream assembly, file I/O, process/pipeline startup, parse/decode, final-picture copies/readback, diagnostics and output extraction/oracle check",
                         "stress_timed": "fixed completed kernel count, per-kernel event synchronization; host unit intervals and CUDA events; work initialization and result validation excluded from stress work duration but included in combined host duration",
                         "comparison": "same calibrated fixed GPU units plus one eight-image inference batch; compare completion time and GPU work rate",
                         "order": "rotate five conditions and reverse on alternate repeats after warmups",
                         "clock": "CLOCK_MONOTONIC shared by Python and helper on Linux",
                         "limitation": "CUDA proxy for shader execution; no graphics API, occupancy, power or hardware execution-overlap claim",
                         "memory_elements": args.elements, "memory_buffer_bytes_each": args.elements * 4,
                         "memory_bytes_per_unit": args.elements * 8,
                         "compute_elements": min(args.elements, 262144), "compute_cycles": args.cycles,
                         "compute_work": "four independent unsigned integer arithmetic chains per element"},
              "modes": {}}

    def save():
        (output / "concurrency.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    with tempfile.TemporaryDirectory(prefix="decoder-concurrency-") as directory:
        try:
            service = CUDAService(args.cuda_helper, args.ptx, bundle, directory, args.elements, args.cycles)
        except RuntimeError as error:
            report["status"] = "cuda-unavailable"
            report["checked_limitation"] = str(error)
            save()
            raise
        try:
            def infer(kind):
                start = time.monotonic_ns()
                if kind == "cuda":
                    details = service.command("INFER", "inference")
                    actual = service.output.read_bytes()
                    if actual != bundle["expected"]:
                        raise AssertionError("CUDA result differs from frozen model oracle")
                else:
                    result = decode_batch(bundle["paths"], "nvdec", width=208, height=112,
                                          pictures_per_job=17, selected_frames=(16,))
                    if result.decoded_frames != 136 or result.copied_frames != 8 or len(result.frames) != 8:
                        raise AssertionError("NVDEC final-picture counts differ")
                    actual = b"".join(extract_final(frames[0]) for frames in result.frames)
                    if actual != bundle["expected"]:
                        raise AssertionError("NVDEC result differs from frozen model oracle")
                    details = {"decoded_pictures": 136, "copied_pictures": 8,
                               "copied_i420_bytes": 8 * 208 * 112 * 3 // 2}
                end = time.monotonic_ns()
                return {"backend": kind, "start_ns": start, "end_ns": end,
                        "host_ms": (end - start) / 1e6, "validation": "pass",
                        "output_sha256": digest(actual), "details": details}

            # Tiny correctness path is explicit; it never supplies measured results.
            if args.smoke_only:
                report["smoke"] = {kind: infer(kind) for kind in ("cuda", "nvdec")}
                for mode in ("compute", "memory"):
                    service.command("STRESS %s 1" % mode, "started")
                    report["smoke"][mode] = service.command("WAIT", "stress")
                report["status"] = "smoke-pass"; save()
                return report

            for mode in ("compute", "memory"):
                # Calibration is a real completed sample, outside measurements.
                service.command("STRESS %s 3" % mode, "started")
                calibration = service.command("WAIT", "stress")
                duration = (calibration["end_ns"] - calibration["start_ns"]) / 1e9
                unit_count = max(3, min(100000, int(math.ceil(args.seconds * 3 / duration))))
                cases = ("gpu-only", "nvdec-only", "cuda-only", "gpu+nvdec", "gpu+cuda")
                entry = {"fixed_gpu_units": unit_count, "calibration": calibration,
                         "samples": {case: [] for case in cases}}
                report["modes"][mode] = entry

                def execute(case):
                    started = time.monotonic_ns()
                    result = {"condition": case}
                    if case.startswith("gpu"):
                        result["ready"] = service.command("STRESS %s %d" % (mode, unit_count), "started")
                    kind = "nvdec" if "nvdec" in case else "cuda" if "cuda" in case else None
                    if kind:
                        result["inference"] = infer(kind)
                    completed = time.monotonic_ns()
                    if case.startswith("gpu"):
                        stress = service.command("WAIT", "stress")
                        completed = time.monotonic_ns()
                        if stress["completed_units"] != unit_count or stress["sample_validation"] != "pass":
                            raise AssertionError("Fixed GPU work was not completed and checked")
                        result["stress"] = stress
                        stress["work_ms"] = (stress["end_ns"] - stress["start_ns"]) / 1e6
                        stress["units_per_second"] = unit_count * 1000 / stress["work_ms"]
                        stress["summed_gpu_event_ms"] = sum(unit[2] for unit in stress["units"])
                        if kind:
                            result["overlap"] = overlap_statistics(stress, result["inference"])
                            if not result["overlap"]["intersecting_units"]:
                                raise AssertionError("Inference and GPU work did not overlap")
                    result["start_ns"] = started; result["end_ns"] = completed
                    result["completed_combined_ms"] = (result["end_ns"] - started) / 1e6
                    return result

                for case in cases:
                    execute(case)
                    print("PASS %s %s warmup" % (mode, case), flush=True)
                for repetition in range(args.repeats):
                    offset = repetition % len(cases)
                    order = list(cases[offset:] + cases[:offset])
                    if repetition % 2:
                        order.reverse()
                    for case in order:
                        sample = execute(case)
                        sample["repetition"] = repetition + 1
                        entry["samples"][case].append(sample)
                        save()
                        print("PASS %s %s run %d: %.3f ms" %
                              (mode, case, repetition + 1, sample["completed_combined_ms"]), flush=True)
                entry["summary"] = summarize(entry)
                save()
            report["status"] = "pass"; save()
            return report
        finally:
            service.close()


def summarize(entry):
    summary = {}
    for case, samples in entry["samples"].items():
        result = {"median_combined_ms": statistics.median(item["completed_combined_ms"] for item in samples)}
        if "inference" in samples[0]:
            result["median_inference_ms"] = statistics.median(item["inference"]["host_ms"] for item in samples)
        if "stress" in samples[0]:
            result["median_gpu_work_ms"] = statistics.median(item["stress"]["work_ms"] for item in samples)
            result["median_gpu_units_per_second"] = statistics.median(item["stress"]["units_per_second"] for item in samples)
        if "overlap" in samples[0]:
            result["min_host_overlap_ms"] = min(item["overlap"]["host_interval_overlap_ms"] for item in samples)
        summary[case] = result
    baseline = summary["gpu-only"]
    for kind in ("cuda", "nvdec"):
        combined = summary["gpu+" + kind]
        combined["gpu_work_rate_fraction_of_alone"] = combined["median_gpu_units_per_second"] / baseline["median_gpu_units_per_second"]
        combined["combined_time_vs_gpu_alone_ratio"] = combined["median_combined_ms"] / baseline["median_combined_ms"]
        combined["inference_slowdown_vs_alone"] = combined["median_inference_ms"] / summary[kind + "-only"]["median_inference_ms"]
        combined["serial_gpu_plus_inference_estimate_ms"] = baseline["median_combined_ms"] + summary[kind + "-only"]["median_inference_ms"]
        combined["speedup_vs_serial_estimate"] = combined["serial_gpu_plus_inference_estimate_ms"] / combined["median_combined_ms"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--cuda-helper", required=True, type=Path)
    parser.add_argument("--ptx", required=True, type=Path)
    parser.add_argument("--helper", type=Path, help="NVDEC helper override")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--cycles", type=int, default=128)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    result = benchmark(args)
    print(json.dumps({"status": result["status"],
                      "summary": {mode: entry.get("summary") for mode, entry in result["modes"].items()}}), flush=True)


if __name__ == "__main__":
    main()
