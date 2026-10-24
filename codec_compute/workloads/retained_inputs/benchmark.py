"""Compare complete multi-output sequences with tuned native CPU execution."""

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import time

spec = importlib.util.spec_from_file_location("_retained_benchmark_runtime", Path(__file__).with_name("runtime.py"))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
MODES = ("native-host", "prepared-host", "prepared-complete")


def trial(corpus, key, mode, repeat):
    w, phases = corpus.workloads[key], {}
    if mode == "cpu":
        corpus.cpu(w, 3)
        accounting = runtime.Accounting()
        start = accounting.snapshot()
        tick = time.perf_counter()
        for index in range(4):
            corpus.cpu(w, index)
        elapsed = time.perf_counter()-tick
        end = accounting.snapshot()
    else:
        tick = time.perf_counter()
        active = corpus.session(w)
        phases["create_ms"] = (time.perf_counter()-tick)*1000
        try:
            tick = time.perf_counter()
            corpus.submit(active, w, 3, mode)
            phases["warmup_ms"] = (time.perf_counter()-tick)*1000
            before = active.stats
            accounting = runtime.Accounting(active._process.pid)
            start = accounting.snapshot()
            tick = time.perf_counter()
            for index in range(4):
                corpus.submit(active, w, index, mode)
            elapsed = time.perf_counter()-tick
            end = accounting.snapshot()
            after = active.stats
            wanted = 0 if mode == "prepared-complete" else 4*w["case"]["output_count"]
            if (after["copied_frames"]-before["copied_frames"] != wanted
                    or after["requests"]-before["requests"] != 4):
                raise AssertionError("Timed counters differ")
            phases.update(stats_before_timing=before, stats_after_timing=after, timed_copied_frames=wanted)
            if mode == "prepared-complete":
                tick = time.perf_counter()
                for index in range(4):
                    corpus.submit(active, w, index, "prepared-host", full=True)
                phases.update(postflight_ms=(time.perf_counter()-tick)*1000, postflight_full_i420_jobs=4)
        finally:
            tick = time.perf_counter()
            active.close()
            phases["close_ms"] = (time.perf_counter()-tick)*1000
        requests = 9 if mode == "prepared-complete" else 5
        copies = 4*w["case"]["costs"]["total_pictures"] if mode == "prepared-complete" else 5*w["case"]["output_count"]
        phases["stats_after_close"] = runtime.counts(active, w["case"], requests, copies)
    return dict(repeat=repeat, fixture_indices=[0,1,2,3], outputs_per_request=w["case"]["output_count"],
                **runtime.measurement(start, end, elapsed, 4), **phases)


def run(root, output, *, cpu_helper, pcm_helper, nvdec_helper, build_info, repeats=5):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Positive repeat count required")
    corpus = runtime.Corpus(root, cpu_helper=cpu_helper, pcm_helper=pcm_helper,
                            nvdec_helper=nvdec_helper, build_info=build_info)
    report = dict(format="codec-compute-retained-benchmark-1", status="running",
                  created_utc=datetime.now(timezone.utc).isoformat(), **corpus.identity,
                  repeats=repeats, requests_per_trial=4, groups={}, order=[], preflight={},
                  contract={
                      "native-host": "Native PCM preparation and exact prefix check, input decode once per complete sequence, all variant predictions, one counted IDR fence, ALL named host outputs and exact comparison. No resets between variants.",
                      "cpu": "Native/unrolled -O3 single-threaded independent supplied-field kernels, returning and checking all compact luma outputs. Reusable contexts allocated before timing; no result cache. Common interpolation planes are not shared across separate variant contexts.",
                      "prepared-host": "Input PCM supplied pre-encoded, but input decode and every requested host result still timed.",
                      "prepared-complete": "Unequal-result diagnostic: input decode and all predictions/fence, zero host results, no GPU consumer. Four NEW full-I420 requests postflight outside timing do not observe earlier unread pixels.",
                      "excluded": "Reusable allocation, pipeline create, one warmup, postflight, final fence drain/close. Per-sequence input decode is included. Setup/warmup/close wall times recorded.",
                      "accounting": "Parent and live worker process CPU, including threads; excludes other processes/IRQs/kernel workqueues. No energy inference.",
                      "scope": "Finite multi-output batch throughput, not interactive retained-reference completion. This measured path uses existing fenced requests; separate streaming probe tests no-reset output availability.",
                      "ordering": "CPU once per workload/size/output-count; six decoder mode/policy variants, five rotated and alternately reversed repeats.",
                      "normalization": "wall_ms_per_job and cpu_ms_per_job refer to the WHOLE output set; per-output values divide by the explicitly recorded output count."})

    def save():
        runtime.common.write_json(output, report)
    try:
        for key, w in corpus.workloads.items():
            with corpus.session(w) as active:
                for index in range(4):
                    corpus.submit(active,w,index,full=True)
            report["preflight"][key] = runtime.counts(active,w["case"],4,4*w["case"]["costs"]["total_pictures"])
            print("PASS retained fenced preflight "+key,flush=True)
            save()
        entries = []
        for group, variants in sorted(corpus.groups.items()):
            report["groups"][group] = dict(cases=variants, outputs_per_request=corpus.workloads[variants["dpb16"]]["case"]["output_count"], modes={})
            entries.append((group,"dpb16","cpu"))
            entries.extend((group,policy,mode) for policy in ("dpb16","tight") for mode in MODES)
        for repeat in range(repeats):
            offset = repeat*11 % len(entries)
            order = entries[offset:]+entries[:offset]
            if repeat%2:
                order = list(reversed(order))
            report["order"].append(order)
            for group, policy, mode in order:
                result = trial(corpus,corpus.groups[group][policy],mode,repeat)
                label = "cpu" if mode=="cpu" else policy+"/"+mode
                record = report["groups"][group]["modes"].setdefault(label,dict(trials=[]))
                record["trials"].append(result)
                record.update(runtime.summarize(record["trials"]))
                count = report["groups"][group]["outputs_per_request"]
                record["wall_ms_per_output"] = record["wall_ms_per_job"]/count
                record["cpu_ms_per_output"] = record["cpu_ms_per_job"]/count
                save()
            print("PASS retained benchmark repeat "+str(repeat+1),flush=True)
        corpus.verify_stable()
        report["status"] = "pass"
        save()
    except Exception as error:
        report.update(status="failed",error=dict(type=type(error).__name__,detail=str(error)))
        save()
        raise
    finally:
        corpus.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    for name in ("cpu-helper","pcm-helper","nvdec-helper","build-info"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    run(args.bundle,args.output,cpu_helper=args.cpu_helper,pcm_helper=args.pcm_helper,
        nvdec_helper=args.nvdec_helper,build_info=args.build_info,repeats=args.repeats)
