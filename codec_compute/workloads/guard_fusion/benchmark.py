#!/usr/bin/env python3
"""Paired slice-layout timings and CPU-time crossover over recurrence depth."""

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_guard_runtime", HERE/"runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def run(root, output, *, cpu_helper, pcm_helper, nvdec_helper, build_info, repeats=5):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    corpus = runtime.Corpus(root, cpu_helper=cpu_helper, pcm_helper=pcm_helper,
                            nvdec_helper=nvdec_helper, build_info=build_info)
    script_hash = runtime.common.digest(Path(__file__).read_bytes())
    if script_hash != corpus.bundle["source_sha256"][str(Path(__file__).resolve().relative_to(runtime.REPOSITORY))]:
        raise AssertionError("Benchmark differs from frozen source")
    report = {"format": "codec-compute-guard-crossover-1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              **corpus.identity, "repeats": repeats, "requests_per_trial": 8,
              "validation": {}, "groups": {}, "trial_order": [],
              "protocol": {"inputs": "Identical changing compact u8 fixtures, same CPU recurrence and all decoded pictures in each paired layout",
                           "cpu": "Previously tuned single-threaded shared-filter native/unrolled C++; output bytes and exact comparison included",
                           "nvdec": "Warm resident pipeline; native input serialization, upload, prediction, fence, selected full-I420 readback and exact comparison included",
                           "excluded": "Loading, reusable scratch/context setup, one warmup and close; decoder lifecycle phases recorded separately",
                           "accounting": "Parent plus live helper process CPU; excludes IRQs, kernel workqueues, other processes; not energy",
                           "interpretation": "Implementation timings, synthetic recurrence; no application-throughput or universal CPU optimality claim"}}

    def save():
        runtime.common.write_json(output, report)

    try:
        for key, w in corpus.workloads.items():
            with corpus.session(key) as active:
                for i in range(8):
                    corpus.submit(active, key, i, full=True)
            report["validation"][key] = {"status": "pass", **runtime.counts(active, w["case"], 8, 8*w["case"]["costs"]["total_pictures"])}
            print("PASS full NVDEC "+key, flush=True)
            save()
        entries = []
        for group in sorted(corpus.groups, key=lambda key: (int(key.split('/')[0]), int(key.rsplit('-', 1)[1]))):
            variants = corpus.groups[group]
            report["groups"][group] = {"cases": variants,
                                       "costs": {v: corpus.workloads[k]["case"]["costs"] for v, k in variants.items()},
                                       "modes": {}}
            entries.extend((group, mode) for mode in ("cpu", "nvdec-baseline", "nvdec-merged"))
        for repeat in range(repeats):
            offset = repeat*7 % len(entries)
            order = entries[offset:]+entries[:offset]
            if repeat % 2:
                order.reverse()
            report["trial_order"].append([{"group": g, "mode": m} for g, m in order])
            for group, mode in order:
                variants = corpus.groups[group]
                key = variants["baseline" if mode == "nvdec-baseline" else "merged"]
                phases = {}
                if mode == "cpu":
                    corpus.cpu(key, 7)
                    accounting = runtime.Accounting()
                    start = accounting.snapshot()
                    tick = time.perf_counter()
                    for i in range(8):
                        corpus.cpu(key, i)
                    elapsed = time.perf_counter()-tick
                    end = accounting.snapshot()
                else:
                    tick = time.perf_counter()
                    active = corpus.session(key)
                    phases["create_ms"] = (time.perf_counter()-tick)*1000
                    try:
                        tick = time.perf_counter()
                        corpus.submit(active, key, 7)
                        phases["warmup_ms"] = (time.perf_counter()-tick)*1000
                        accounting = runtime.Accounting(active._process.pid)
                        start = accounting.snapshot()
                        tick = time.perf_counter()
                        for i in range(8):
                            corpus.submit(active, key, i)
                        elapsed = time.perf_counter()-tick
                        end = accounting.snapshot()
                    finally:
                        tick = time.perf_counter()
                        active.close()
                        phases["close_ms"] = (time.perf_counter()-tick)*1000
                    phases["stats_including_warmup"] = runtime.counts(active, corpus.workloads[key]["case"], 9, 9)
                trial = {"repeat": repeat, "fixture_indices": list(range(8)),
                         **runtime.measurement(start, end, elapsed, 8), **phases}
                record = report["groups"][group]["modes"].setdefault(mode, {"trials": []})
                record["trials"].append(trial)
                record.update(runtime.summarize(record["trials"]))
                save()
            print("PASS crossover repeat "+str(repeat+1), flush=True)
        corpus.verify_stable()
        if runtime.common.digest(Path(__file__).read_bytes()) != script_hash:
            raise AssertionError("Benchmark script changed during execution")
        report["status"] = "pass"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "detail": str(error)}
        save()
        raise
    finally:
        corpus.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("cpu-helper", "pcm-helper", "nvdec-helper", "build-info"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.bundle, args.output, cpu_helper=args.cpu_helper, pcm_helper=args.pcm_helper,
        nvdec_helper=args.nvdec_helper, build_info=args.build_info, repeats=args.repeats)
