#!/usr/bin/env python3
"""Same-host CPU/NVDEC comparisons. Python3.8+, no compiler at runtime."""

import argparse
import ctypes
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
resident = module("_comparison_resident", HERE.parent/"resident/benchmark.py")
native = module("_comparison_cpu", HERE/"native.py")
common = resident.common
from decoder_core.session import NVDECSession
from decoder_core.pcm import NativePCMWriter
from decoder_core.decode import decode_batch


class Accounting:
    """Process-accounted CPU, not system-wide CPU or energy measurement."""
    def __init__(self, pid=None):
        self.clock = None
        self.pid = pid
        self.ticks = os.sysconf("SC_CLK_TCK")
        if pid is not None:
            libc = ctypes.CDLL(None)
            function = libc.clock_getcpuclockid
            function.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            clock = ctypes.c_int()
            code = function(pid, ctypes.byref(clock))
            if code:
                raise OSError(code, "Cannot read live worker process CPU clock")
            self.clock = clock.value

    def snapshot(self, *, children=False):
        own = resource.getrusage(resource.RUSAGE_SELF)
        result = {"parent_cpu_s": time.process_time(), "parent_user_s": own.ru_utime,
                  "parent_system_s": own.ru_stime}
        if self.clock is not None:
            result["worker_cpu_s"] = time.clock_gettime_ns(self.clock)/1e9
            fields = Path("/proc/{}/stat".format(self.pid)).read_text().rsplit(")", 1)[1].split()
            result["worker_user_s"] = int(fields[11])/self.ticks
            result["worker_system_s"] = int(fields[12])/self.ticks
        elif children:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            result.update(worker_cpu_s=usage.ru_utime+usage.ru_stime,
                          worker_user_s=usage.ru_utime, worker_system_s=usage.ru_stime)
        return result


def measurement(start, end, elapsed, requests):
    result = {key: end[key]-value for key, value in start.items()}
    if any(value < -1e-6 for value in result.values()):
        raise RuntimeError("CPU accounting went backwards")
    result.update(elapsed_ms=elapsed*1000, requests=requests)
    result["process_cpu_ms"] = 1000*(result["parent_cpu_s"]+result.get("worker_cpu_s", 0))
    return result


def summarize(trials):
    wall = [trial["elapsed_ms"]/trial["requests"] for trial in trials]
    cpu = [trial["process_cpu_ms"]/trial["requests"] for trial in trials]
    return {"wall_ms_per_job": statistics.median(wall), "wall_range_ms": [min(wall), max(wall)],
            "cpu_ms_per_job": statistics.median(cpu), "cpu_range_ms": [min(cpu), max(cpu)]}


def counts(active, case, jobs, copies):
    actual = active.stats
    expected = {"requests": jobs, "submitted_frames": jobs*(case["costs"]["total_pictures"]+1),
                "decoded_frames": jobs*(case["costs"]["total_pictures"]+1),
                "copied_frames": copies, "fence_pictures": jobs, "decoded_fences": jobs,
                "pending_fences": 0, "pipeline_starts": 1, "fence_bytes": jobs*case["fence"]["bytes"]}
    if any(actual[k] != v for k, v in expected.items()):
        raise AssertionError("Resident counters differ: "+repr(actual))
    return actual


def run(root, output, *, cpu_helper, pcm_helper, nvdec_helper, planes_helper, build_info, repeats=5, requests=64):
    root, output = Path(root).resolve(), Path(output).resolve()
    helpers = {name: Path(value).resolve() for name, value in
               (("cpu", cpu_helper), ("pcm", pcm_helper), ("nvdec", nvdec_helper), ("cpu_planes", planes_helper))}
    if repeats < 1 or requests < 8 or requests % 8:
        raise ValueError("Positive repeats and a positive multiple of eight requests required")
    data, bundle, workloads = resident.load(root)
    repository = HERE.parents[2]
    if bundle.get("experiment") != "cpu-comparison":
        raise ValueError("Expected CPU comparison bundle")
    for name in ("decoder_core/session.py", "decoder_core/decode.py", "decoder_core/bitstream.py",
                 "decoder_core/pcm.py", "decoder_core/pcm_native.cpp", "decoder_core/nvdec_decode.cpp",
                 "codec_compute/workloads/resident/benchmark.py", "codec_compute/workloads/temporal_benchmark.py",
                 "codec_compute/workloads/cpu_comparison/native.py", "codec_compute/workloads/cpu_comparison/cpu.cpp",
                 "codec_compute/workloads/cpu_comparison/cpu_planes.cpp",
                 "codec_compute/workloads/cpu_comparison/benchmark.py"):
        if common.digest((repository/name).read_bytes()) != bundle["source_sha256"][name]:
            raise AssertionError("Runtime source differs: "+name)
    build = json.loads(Path(build_info).read_text())
    for key in ("cpu", "pcm", "cpu_planes"):
        if build["helper_sha256"][key] != common.digest(helpers[key].read_bytes()):
            raise AssertionError("Native binary differs from its build record")
    for source, checksum in build["source_sha256"].items():
        if bundle["source_sha256"][source] != checksum:
            raise AssertionError("Native build source differs from prepared bundle")
    Writer = common.writer_class(repository/"decoder_core/bitstream.py")
    os.environ["H264_NVDEC_DECODER"] = str(helpers["nvdec"])
    report = {"format": "codec-compute-cpu-comparison-1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": bundle["source_commit"],
              "source_sha256": bundle["source_sha256"], "bundle_sha256": common.digest(data),
              "helper_sha256": {key: common.digest(path.read_bytes()) for key, path in helpers.items()},
              "native_build": build, "fixtures_per_case": bundle["jobs"],
              "python": sys.version.split()[0], "repeats": repeats, "short_requests": requests,
              "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"), "cases": {},
              "protocol": {
                  "cpu": "Single-threaded optimized C++ via ctypes, preallocated scratch/output, compact host u8 inputs, returned bytes and exact equality checked every request; no memoization",
                  "cpu_planes": "Same recurrence, specialized to its fixed two fields; shared half-sample calculations eliminate redundant filtering; all rounds and clipping retained",
                  "nvdec_preencoded": "Warm live pipeline, frozen prepared access units; fresh input picture uploads still decoded each request; exact returned I420 checked",
                  "nvdec_native": "Same warm path, native I_PCM serialization of compact host inputs, existing Writer limits plus exact prefix equality checked; host image inputs and host result",
                  "batch": "Finite 64-job batch, no completion fences; input assembly/file I/O, process/startup, decode, selected readback, equality and EOS/shutdown included; prepared-input and native-host variants",
                  "warm": "Context setup/reusable scratch, one full warmup and final close excluded from warm timings; per-request allocations remain timed; decoder lifecycle phases recorded separately",
                  "cpu_accounting": "Parent process CPU plus live worker process CPU clock; finite batches use reaped child CPU. Includes all threads in those processes. Excludes other processes, IRQs and kernel workqueues; not energy",
                  "inputs": "Prepared changing scenes/noise/extremes cycled in order (fixtures_per_case records count); warm cache behavior, fixed motion fields. Chains retain intermediates; only final result copied",
                  "limits": "Implementation comparison on one host, not isolated hardware throughput. CPU single-threaded, no hand-written SIMD; -O3 compiler vectorization. No universal claim or energy/power measurement"}}

    def save():
        common.write_json(output, report)

    def session(w):
        return NVDECSession(w["case"]["coded_width"], w["case"]["coded_height"], helper=helpers["nvdec"], timeout=60)

    def encode(w, job):
        prefix = tuple(w["encoder"].input_luma(job["compact"][name], i)
                       for i, name in enumerate(w["case"]["input_names"]))
        if prefix != job["prefix"]:
            raise AssertionError("Native PCM input bytes differ")
        return prefix+w["tail"]

    def submit(active, w, job, host=False, full=False):
        selection = None if full else w["selection"]
        actual = active.submit(encode(w, job) if host else job["units"], selection, fence=w["fence"])
        expected = job["frames"] if full else tuple(job["frames"][i] for i in selection)
        if actual != expected:
            raise AssertionError("NVDEC full returned I420 differs")

    for key, w in workloads.items():
        case = w["case"]
        params = dict(case["writer"])
        params["reference_slots"] = {int(k): v for k, v in params["reference_slots"].items()}
        w["encoder"] = NativePCMWriter(Writer(**params), helper=helpers["pcm"])
        w["selection"] = tuple(sorted(set(case["output_frames"].values())))
        w["cpu"] = native.CPU(helpers["cpu"], case["size"], case["cpu"])
        if case["cpu"]["kind"] == 4:
            w["cpu_planes"] = native.CPU(helpers["cpu_planes"], case["size"], case["cpu"])
        for job, record in zip(w["jobs"], case["jobs"]):
            job["compact"] = {name: common.checked(root, value) for name, value in record["inputs"].items()}
            job["output"] = common.checked(root, record["outputs"]["output"])
            if common.checked(root, record["ordinary"]) != b"".join(job["units"]):
                raise AssertionError("Ordinary batch stream differs from access units")
            job["ordinary"] = root/record["ordinary"]["path"]
            if w["cpu"].run(*(job["compact"][name] for name in case["input_names"])) != job["output"]:
                raise AssertionError("CPU output differs")
            if "cpu_planes" in w and w["cpu_planes"].run(job["compact"]["image"]) != job["output"]:
                raise AssertionError("Specialized CPU recurrence differs")
            encode(w, job)
        active = session(w)
        with active:
            for job in w["jobs"]:
                submit(active, w, job, host=True, full=True)
        stat = counts(active, case, len(w["jobs"]), len(w["jobs"])*case["costs"]["total_pictures"])
        report["cases"][key] = {"size": case["size"], "cpu_spec": case["cpu"], "costs": case["costs"],
                               "validation": {"status": "pass", **stat}, "modes": {}}
        print("PASS CPU/native-upload/full NVDEC "+key, flush=True)
        save()

    entries = [(key, mode) for key, w in workloads.items()
               for mode in ("cpu", "native-serialization", "nvdec-preencoded", "nvdec-native")]
    entries += [(key, "cpu-planes") for key, w in workloads.items() if "cpu_planes" in w]
    entries += [(key, mode) for key, w in workloads.items()
                if w["case"]["size"] >= 256 and w["case"]["cpu"]["kind"] in (1, 2, 3)
                for mode in ("batch-preencoded", "batch-native")]
    for repeat in range(repeats):
        offset = repeat*11 % len(entries)
        for key, mode in entries[offset:]+entries[:offset]:
            w = workloads[key]
            number = 64 if mode.startswith("batch") else (8 if w["case"]["cpu"]["kind"] == 4 else requests)
            jobs = [w["jobs"][i % len(w["jobs"])] for i in range(number)]
            phases = {}
            if mode.startswith("nvdec"):
                tick = time.perf_counter()
                active = session(w)
                phases["create_ms"] = (time.perf_counter()-tick)*1000
                try:
                    tick = time.perf_counter()
                    submit(active, w, jobs[-1], host=mode == "nvdec-native")
                    phases["warmup_ms"] = (time.perf_counter()-tick)*1000
                    accounting = Accounting(active._process.pid)
                    start = accounting.snapshot()
                    tick = time.perf_counter()
                    for job in jobs:
                        submit(active, w, job, host=mode == "nvdec-native")
                    elapsed = time.perf_counter()-tick
                    end = accounting.snapshot()
                finally:
                    tick = time.perf_counter()
                    active.close()
                    phases["close_ms"] = (time.perf_counter()-tick)*1000
                phases["stats_including_warmup"] = counts(active, w["case"], number+1, number+1)
            elif mode.startswith("batch"):
                accounting = Accounting()
                start = accounting.snapshot(children=True)
                tick = time.perf_counter()
                with tempfile.TemporaryDirectory(prefix="cpu-comparison-batch-") as temp:
                    paths = []
                    for i, job in enumerate(jobs):
                        if mode == "batch-native":
                            path = Path(temp)/(str(i)+".h264")
                            path.write_bytes(b"".join(encode(w, job)))
                        else:
                            path = job["ordinary"]
                        paths.append(path)
                    result = decode_batch(paths, width=w["case"]["coded_width"], height=w["case"]["coded_height"],
                                          pictures_per_job=w["case"]["costs"]["total_pictures"], selected_frames=w["selection"])
                    expected = tuple(tuple(job["frames"][i] for i in w["selection"]) for job in jobs)
                    if result.frames != expected:
                        raise AssertionError("Finite batch differs")
                elapsed = time.perf_counter()-tick
                end = accounting.snapshot(children=True)
                phases["decoded_frames"] = result.decoded_frames
                phases["copied_frames"] = result.copied_frames
            else:
                def execute(job):
                    if mode == "native-serialization":
                        encode(w, job)
                    elif w["cpu_planes" if mode == "cpu-planes" else "cpu"].run(*(job["compact"][name] for name in w["case"]["input_names"])) != job["output"]:
                        raise AssertionError("Timed CPU output differs")
                execute(jobs[-1])
                accounting = Accounting()
                start = accounting.snapshot()
                tick = time.perf_counter()
                for job in jobs:
                    execute(job)
                elapsed = time.perf_counter()-tick
                end = accounting.snapshot()
            trial = {**measurement(start, end, elapsed, number), **phases}
            record = report["cases"][key]["modes"].setdefault(mode, {"trials": []})
            record["trials"].append(trial)
            record.update(summarize(record["trials"]))
            save()
        print("PASS timing repeat "+str(repeat+1), flush=True)
    for w in workloads.values():
        w["cpu"].close()
        if "cpu_planes" in w:
            w["cpu_planes"].close()
    report["status"] = "pass"
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-helper", type=Path, required=True)
    parser.add_argument("--planes-helper", type=Path, required=True)
    parser.add_argument("--pcm-helper", type=Path, required=True)
    parser.add_argument("--nvdec-helper", type=Path, required=True)
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--requests", type=int, default=64)
    args = parser.parse_args()
    run(args.bundle, args.output, cpu_helper=args.cpu_helper, pcm_helper=args.pcm_helper,
        nvdec_helper=args.nvdec_helper, planes_helper=args.planes_helper, build_info=args.build_info, repeats=args.repeats, requests=args.requests)
