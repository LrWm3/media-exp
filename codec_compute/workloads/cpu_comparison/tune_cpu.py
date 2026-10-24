#!/usr/bin/env python3
"""CPU-only flag follow-up on frozen eight-fixture recurrence exports.

Python 3.8+, standard library plus the existing portable loader/ctypes adapter.
Run this file directly; no compiler IR or decoder process is imported/executed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


resident = module("_tuning_resident", HERE.parent/"resident/benchmark.py")
native = module("_tuning_native", HERE/"native.py")
common = resident.common


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_flag(architecture):
    if architecture.lower() in ("aarch64", "arm64"):
        return "-mcpu=native"
    if architecture.lower() in ("x86_64", "amd64", "i386", "i486", "i586", "i686"):
        return "-march=native"
    raise ValueError("CPU tuning supports only AArch64 and x86 architectures")


def summarize(trials):
    wall = [trial["elapsed_ms"]/trial["requests"] for trial in trials]
    cpu = [trial["process_cpu_ms"]/trial["requests"] for trial in trials]
    return {"wall_ms_per_job": statistics.median(wall), "wall_range_ms": [min(wall), max(wall)],
            "cpu_ms_per_job": statistics.median(cpu), "cpu_range_ms": [min(cpu), max(cpu)]}


def run(bundle_path, output, *, compiler="c++", build_dir=None, repeats=5):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    architecture = platform.machine().lower()
    tuning_flag = native_flag(architecture)
    root, output = Path(bundle_path).resolve(), Path(output).resolve()
    build_dir = Path(build_dir).resolve() if build_dir else output.parent/"cpu-tuning-build"
    if output == root/"bundle.json":
        raise ValueError("Output must not replace the frozen bundle")
    compiler_path = shutil.which(compiler)
    if compiler_path is None:
        raise ValueError("Requested C++ compiler is unavailable")
    compiler_path = Path(compiler_path).resolve()
    compiler_hash = digest(compiler_path)
    version = subprocess.check_output([str(compiler_path), "--version"], universal_newlines=True).splitlines()[0]
    data, bundle, workloads = resident.load(root)
    sizes = bundle.get("sizes")
    if (bundle.get("experiment") != "cpu-comparison" or bundle.get("jobs") != 8 or "selection" in bundle
            or not isinstance(sizes, list) or not sizes
            or any(type(size) is not int or size not in (128, 256, 512) for size in sizes)
            or len(sizes) != len(set(sizes))):
        raise ValueError("Expected an original eight-fixture CPU comparison corpus")
    selected = {"{}/resident-chain-{}/fused".format(size, rounds) for size in sizes for rounds in (16, 64)}
    if not selected <= set(workloads):
        raise ValueError("Every corpus size must provide both 16- and 64-round chains")
    sources = ("codec_compute/workloads/cpu_comparison/cpu.cpp",
               "codec_compute/workloads/cpu_comparison/cpu_planes.cpp",
               "codec_compute/workloads/cpu_comparison/native.py",
               "codec_compute/workloads/resident/benchmark.py",
               "codec_compute/workloads/temporal_benchmark.py",
               "decoder_core/session.py", "decoder_core/decode.py")
    checked_sources = {name: digest(REPOSITORY/name) for name in sources}
    for name, checksum in checked_sources.items():
        if checksum != bundle["source_sha256"].get(name):
            raise AssertionError("Current source differs from frozen export: "+name)
    script_hash = digest(__file__)
    report = {"format": "codec-compute-cpu-tuning-1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": bundle["source_commit"], "bundle_sha256": common.digest(data),
              "source_sha256": checked_sources,
              "script": {"path": str(Path(__file__).resolve().relative_to(REPOSITORY)), "sha256": script_hash},
              "compiler": {"version": version, "executable_sha256": compiler_hash},
              "architecture": architecture, "python": sys.version.split()[0],
              "repeats": repeats, "requests_per_trial": 8, "warmup_requests_per_trial": 1,
              "builds": {}, "cases": {}, "trial_order": [],
              "protocol": {
                  "scope": "CPU-only follow-up; no new decoder, compilation-of-IR, or hardware timing claim",
                  "kernels": "Frozen generic and shared-filter C++ kernels, single-threaded, identical fixed fields and recurrence rounds",
                  "timed": "Eight changing compact host inputs, ctypes calls, native computation, output-byte allocation/copy, and exact equality on every call",
                  "excluded": "C++ builds, artifact loading/hashing, reusable context and scratch allocation, initial full-fixture checks, one warmup call per trial, and final context destruction",
                  "rotation": "Deterministic cyclic starting positions across case/variant pairs; odd repeats reverse the order",
                  "cpu_accounting": "time.process_time_ns for this process, including its threads; excludes compiler children, other processes, interrupts and kernel workqueues; not energy",
                  "interpretation": "Finite warm-cache trials; architecture-native flags may produce nonportable binaries; compare measured ranges and raw samples"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    contexts, jobs, artifacts, binaries = {}, {}, {}, {}

    def save():
        common.write_json(output, report)

    def clean(text):
        return text.replace(str(REPOSITORY), "REPOSITORY").replace(str(build_dir), "BUILD")

    def checked(item):
        value = common.checked(root, item)
        if item["path"] in artifacts and artifacts[item["path"]] != item:
            raise AssertionError("Conflicting artifact records")
        artifacts[item["path"]] = item
        return value

    def execute(key, variant, index):
        inputs, expected = jobs[key][index]
        if contexts[key, variant].run(*inputs) != expected:
            raise AssertionError("CPU result differs: {} {} fixture {}".format(key, variant, index))

    try:
        save()
        diagnostics = (["-Rpass=loop-vectorize", "-Rpass=slp-vectorizer"] if "clang" in version.lower()
                       else ["-fopt-info-vec-optimized"])
        modes = (("portable", []), ("native", [tuning_flag]),
                 ("native-unroll", [tuning_flag, "-funroll-loops"]))
        for implementation, source in (("generic", sources[0]), ("planes", sources[1])):
            for mode, extra in modes:
                variant = implementation+"-"+mode
                target = build_dir/("lib"+variant+".so")
                flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared", "-Wall", "-Wextra"]+extra+diagnostics
                completed = subprocess.run([str(compiler_path), *flags, source, "-o", str(target)],
                                           cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           universal_newlines=True, timeout=180,
                                           env=dict(os.environ, LC_ALL="C", GCC_COLORS=""))
                record = {"implementation": implementation, "mode": mode, "source": source,
                          "source_sha256": checked_sources[source], "flags": flags,
                          "binary_file": target.name, "returncode": completed.returncode,
                          "stdout": clean(completed.stdout), "stderr": clean(completed.stderr)}
                report["builds"][variant] = record
                if completed.returncode:
                    save()
                    raise RuntimeError("Compiler rejected variant "+variant+"; inspect recorded build diagnostics")
                record["binary_sha256"] = digest(target)
                binaries[variant] = target
                save()
        for key in sorted(selected):
            case = workloads[key]["case"]
            rounds = int(key.split("/")[1].rsplit("-", 1)[1])
            if (case["cpu"]["kind"] != 4 or case["cpu"]["rounds"] != rounds
                    or case["input_names"] != ["image"] or set(case["output_frames"]) != {"output"}
                    or str(case["size"]) != key.split("/")[0]):
                raise AssertionError("Recurrence case metadata differs")
            checked(case["manifest"])
            jobs[key] = [(tuple(checked(job["inputs"][name]) for name in case["input_names"]),
                          checked(job["outputs"]["output"])) for job in case["jobs"]]
            report["cases"][key] = {"size": case["size"], "cpu_spec": case["cpu"],
                                    "manifest": case["manifest"], "fixtures": len(jobs[key]), "variants": {}}
            for variant, binary in binaries.items():
                contexts[key, variant] = native.CPU(binary, case["size"], case["cpu"])
                for index in range(8):
                    execute(key, variant, index)
                report["cases"][key]["variants"][variant] = {"validated_fixtures": 8, "trials": []}
            save()
        entries = [(key, variant) for key in sorted(selected) for variant in binaries]
        for repeat in range(repeats):
            offset = repeat*5 % len(entries)
            order = entries[offset:]+entries[:offset]
            if repeat % 2:
                order.reverse()
            report["trial_order"].append([{"case": key, "variant": variant} for key, variant in order])
            for key, variant in order:
                execute(key, variant, 7)
                cpu_start = time.process_time_ns()
                wall_start = time.perf_counter_ns()
                for index in range(8):
                    execute(key, variant, index)
                elapsed = time.perf_counter_ns()-wall_start
                process = time.process_time_ns()-cpu_start
                trial = {"repeat": repeat, "requests": 8, "fixture_indices": list(range(8)),
                         "elapsed_ms": elapsed/1e6, "process_cpu_ms": process/1e6}
                record = report["cases"][key]["variants"][variant]
                record["trials"].append(trial)
                record.update(summarize(record["trials"]))
                save()
            print("PASS CPU tuning repeat {}".format(repeat+1), flush=True)
        if digest(root/"bundle.json") != report["bundle_sha256"] or digest(__file__) != script_hash:
            raise AssertionError("Bundle or tuning script changed during execution")
        for name, checksum in checked_sources.items():
            if digest(REPOSITORY/name) != checksum:
                raise AssertionError("Source changed during execution: "+name)
        if digest(compiler_path) != compiler_hash:
            raise AssertionError("Compiler changed during execution")
        for variant, binary in binaries.items():
            if digest(binary) != report["builds"][variant]["binary_sha256"]:
                raise AssertionError("Native binary changed during execution")
        for item in artifacts.values():
            common.checked(root, item)
        report["validation"] = {"cases": len(selected), "variants": len(binaries),
                                "initial_exact_calls": len(entries)*8,
                                "warmup_exact_calls": len(entries)*repeats,
                                "timed_exact_calls": len(entries)*repeats*8,
                                "final_artifact_hash_checks": len(artifacts)}
        report["status"] = "pass"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "detail": clean(str(error))}
        raise
    finally:
        for context in contexts.values():
            context.close()
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.bundle, args.output, compiler=args.compiler, build_dir=args.build_dir, repeats=args.repeats)
