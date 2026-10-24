#!/usr/bin/env python3
"""Separate host-result costs from an explicitly unequal completion-only diagnostic."""

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_interface_runtime", HERE/"runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
MODES = ("native-host", "prepared-host", "prepared-complete")


def ordered_trials(groups, repeat):
    entries = []
    for group in sorted(groups):
        entries.append((group, "rows-1", "cpu"))
        entries.extend((group, variant, mode) for variant in runtime.LAYOUTS for mode in MODES)
    offset = repeat*11 % len(entries)
    entries = entries[offset:]+entries[:offset]
    return list(reversed(entries)) if repeat % 2 else entries


def trial(corpus, key, mode, repeat):
    """Four changing requests; decoder setup, warmup, audit and drain excluded."""
    case, phases = corpus.workloads[key]["case"], {}
    if mode == "cpu":
        corpus.cpu(key, 3)
        accounting = runtime.Accounting()
        start = accounting.snapshot()
        tick = time.perf_counter()
        for index in range(4):
            corpus.cpu(key, index)
        elapsed = time.perf_counter()-tick
        end = accounting.snapshot()
    else:
        if mode not in MODES:
            raise ValueError("Unknown timing mode")
        tick = time.perf_counter()
        active = corpus.session(key)
        phases["create_ms"] = (time.perf_counter()-tick)*1000
        try:
            def request(index):
                if mode == "prepared-complete":
                    corpus.complete(active, key, index)
                else:
                    corpus.submit(active, key, index, mode=mode)
            tick = time.perf_counter()
            request(3)
            phases["warmup_ms"] = (time.perf_counter()-tick)*1000
            before = dict(active.stats)
            accounting = runtime.Accounting(active._process.pid)
            start = accounting.snapshot()
            tick = time.perf_counter()
            for index in range(4):
                request(index)
            elapsed = time.perf_counter()-tick
            end = accounting.snapshot()
            after = dict(active.stats)
            copied = after["copied_frames"]-before["copied_frames"]
            expected_copies = 0 if mode == "prepared-complete" else 4*len(corpus.workloads[key]["selection"])
            if copied != expected_copies or after["requests"]-before["requests"] != 4:
                raise AssertionError("Timed request/readback counters differ")
            phases["timed_copied_frames"] = copied
            phases["stats_before_timing"] = before
            phases["stats_after_timing_before_final_drain"] = after
            if mode == "prepared-complete":
                # These are new decoded requests, outside timing. They verify
                # the frozen stream and live pipeline, not inaccessible pixels
                # from the earlier completion-only requests.
                tick = time.perf_counter()
                for index in range(4):
                    corpus.submit(active, key, index, mode="prepared-host", full=True)
                phases["postflight_ms"] = (time.perf_counter()-tick)*1000
                phases["postflight_full_i420_jobs"] = 4
        finally:
            tick = time.perf_counter()
            active.close()
            phases["close_ms"] = (time.perf_counter()-tick)*1000
        requests = 9 if mode == "prepared-complete" else 5
        copies = 4*case["costs"]["total_pictures"] if mode == "prepared-complete" else 5*len(corpus.workloads[key]["selection"])
        phases["stats_after_close_including_warmup_and_postflight"] = runtime.counts(active, case, requests, copies)
    return {"repeat": repeat, "fixture_indices": [0, 1, 2, 3],
            **runtime.measurement(start, end, elapsed, 4), **phases}


def run(root, output, *, cpu_helper, planes_helper, pcm_helper, nvdec_helper, build_info, repeats=5):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Repeats must be a positive integer")
    corpus = runtime.Corpus(root, cpu_helper=cpu_helper, planes_helper=planes_helper,
                            pcm_helper=pcm_helper, nvdec_helper=nvdec_helper, build_info=build_info)
    script_hash = runtime.common.digest(Path(__file__).read_bytes())
    report = {"format": "codec-compute-interface-limits-1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              **corpus.identity, "repeats": repeats, "requests_per_trial": 4,
              "validation": {"full_nvdec_preflight": {}, "cpu_and_native_upload_checks_per_case": 4},
              "groups": {}, "trial_order": [],
              "protocol": {
                  "cpu": "Native/unrolled -O3 single-threaded C++; generic exact kernels for control/atlas/interpolation, shared-plane fixed-field recurrence kernel for chains; compact host inputs, result bytes and exact comparison timed",
                  "native-host": "Fresh native PCM serialization of compact host inputs; exact prefix check, input decode, prediction, completion fence, selected full-I420 host readback and exact comparison timed",
                  "prepared-host": "Prepared compressed input access units still decoded each request; prediction, fence, selected full-I420 host readback and exact comparison timed",
                  "prepared-complete": "Unequal-interface diagnostic omitting host work: prepared compressed inputs still submitted and decoded, all real pictures awaited, zero returned host results. No readback, output pixel checks, GPU consumer, surface retention or surface handoff in timed requests. Not an application-throughput win",
                  "validation": "All four fixtures and every real I420 picture checked before timing in every layout. Every completion-only trial is followed by four NEW full-I420 decode/check requests outside timing; those checks do not observe pixels of preceding completion-only requests",
                  "excluded": "Source/artifact loading, allocation of reusable CPU scratch and PCM templates, decoder setup, one warmup, completion-only postflight and final fence drain/close; lifecycle times reported",
                  "ordering": "Five rotated repeats by default; four changing fixtures per trial; CPU measured once per workload per repeat, each decoder interface per layout",
                  "accounting": "Parent plus live helper process CPU, including their threads; excludes IRQs, kernel workqueues, other processes. No energy/power inference",
                  "scope": "Same-host implementation and interface sensitivity, warm caches and static motion fields; no universal CPU optimality or direct GPU-surface import claim"}}

    def save():
        runtime.common.write_json(output, report)

    try:
        name = str(Path(__file__).resolve().relative_to(runtime.REPOSITORY))
        if script_hash != corpus.bundle["source_sha256"].get(name):
            raise AssertionError("Benchmark differs from frozen source")
        for key, w in corpus.workloads.items():
            with corpus.session(key) as active:
                if active.protocol_version < 2:
                    raise RuntimeError("This experiment requires a protocol v2 worker")
                for index in range(4):
                    corpus.submit(active, key, index, mode="native-host", full=True)
            report["validation"]["full_nvdec_preflight"][key] = {
                "status": "pass", **runtime.counts(active, w["case"], 4, 4*w["case"]["costs"]["total_pictures"])}
            print("PASS full NVDEC "+key, flush=True)
            save()
        for group, variants in corpus.groups.items():
            report["groups"][group] = {"cases": variants,
                                       "costs": {v: corpus.workloads[k]["case"]["costs"] for v, k in variants.items()},
                                       "modes": {}}
        for repeat in range(repeats):
            order = ordered_trials(corpus.groups, repeat)
            report["trial_order"].append([{"group": g, "layout": v, "mode": m} for g, v, m in order])
            for group, variant, mode in order:
                key = corpus.groups[group][variant]
                result = trial(corpus, key, mode, repeat)
                label = "cpu" if mode == "cpu" else variant+"/"+mode
                record = report["groups"][group]["modes"].setdefault(label, {"trials": []})
                record["trials"].append(result)
                record.update(runtime.summarize(record["trials"]))
                save()
            print("PASS interface repeat "+str(repeat+1), flush=True)
        corpus.verify_stable()
        if runtime.common.digest(Path(__file__).read_bytes()) != script_hash:
            raise AssertionError("Benchmark changed during execution")
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
    for name in ("cpu-helper", "planes-helper", "pcm-helper", "nvdec-helper", "build-info"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.bundle, args.output, cpu_helper=args.cpu_helper, planes_helper=args.planes_helper,
        pcm_helper=args.pcm_helper, nvdec_helper=args.nvdec_helper, build_info=args.build_info, repeats=args.repeats)
