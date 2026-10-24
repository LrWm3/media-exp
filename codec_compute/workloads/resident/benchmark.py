#!/usr/bin/env python3
"""Portable Python3.8 exact benchmark: fresh versus reused NVDEC sessions."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

COMMON = Path(__file__).resolve().parents[1]/"temporal_benchmark.py"
spec = importlib.util.spec_from_file_location("_resident_common", COMMON)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
from decoder_core.session import NVDECSession


def load(root):
    data = (root/"bundle.json").read_bytes()
    bundle = json.loads(data)
    if bundle.get("format") != "codec-compute-resident-1" or bundle.get("status") != "pass":
        raise ValueError("Expected passing resident experiment bundle")
    workloads = {}
    for key, case in bundle["cases"].items():
        manifest = json.loads(common.checked(root, case["manifest"]))
        shape = common.geometry(manifest)
        if (case["size"] != shape["width"] or case["size"] != shape["height"]
                or case["coded_width"] != shape["coded_width"] or case["coded_height"] != shape["coded_height"]
                or case["frame_bytes"] != shape["frame_bytes"] or case["output_frames"] != shape["output_frames"]
                or case["level_idc"] != manifest["level_idc"] or case["costs"] != manifest["costs"]):
            raise AssertionError("Case geometry, costs or level differs from manifest")
        tail = tuple(common.checked(root, item) for item in case["prediction_units"])
        fence = common.checked(root, case["fence"])
        if len(tail) != case["costs"]["computation_pictures"] or sum(map(len, tail)) != case["costs"]["prediction_bytes"]:
            raise AssertionError("Prediction units differ from manifest")
        jobs = []
        for job in case["jobs"]:
            inputs = {name: common.checked(root, record) for name, record in job["inputs"].items()}
            if set(inputs) != set(case["input_names"]) or any(len(p) != case["size"]**2 for p in inputs.values()):
                raise AssertionError("Input names or sizes differ")
            prefix = tuple(common.checked(root, record) for record in job["units"])
            units = prefix+tail
            if len(prefix) != len(case["input_names"]) or len(units) != shape["pictures_per_job"]:
                raise AssertionError("Input or total picture count differs")
            if common.checked(root, job["stream"]) != b"".join(units)+fence:
                raise AssertionError("Saved stream differs from framed access units and fence")
            raw = common.checked(root, job["expected"])
            frame_bytes = shape["frame_bytes"]
            if len(raw) != (len(units)+1)*frame_bytes or raw[-frame_bytes:] != bytes([128])*frame_bytes:
                raise AssertionError("Expected I420 count or neutral fence differs")
            frames = tuple(raw[i:i+frame_bytes] for i in range(0, len(raw)-frame_bytes, frame_bytes))
            common.validate_guards(frames, shape)
            for index, name in enumerate(case["input_names"]):
                if common.extract(frames[index], shape) != inputs[name]:
                    raise AssertionError("Decoded reference input differs from input plane")
            if set(job["outputs"]) != set(case["output_frames"]):
                raise AssertionError("Named output set differs")
            for name, index in case["output_frames"].items():
                if common.extract(frames[index], shape) != common.checked(root, job["outputs"][name]):
                    raise AssertionError("Named output differs from full decoded reference")
            size = case["size"]
            boards = {name: [list(p[y*size:(y+1)*size]) for y in range(size)] for name, p in inputs.items()}
            jobs.append({"units": units, "prefix": prefix, "boards": boards, "frames": frames})
        if len(jobs) != bundle["jobs"]:
            raise AssertionError("Missing prepared jobs")
        workloads[key] = {"case": case, "shape": shape, "tail": tail, "fence": fence, "jobs": jobs}
    return data, bundle, workloads


def summarize(samples, jobs):
    return {"samples_ms": samples, "median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples),
            "amortized_ms_per_job": statistics.median(samples)/jobs}


def run(root, output, *, helper, repeats=5, cases=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    root, output, helper = Path(root).resolve(), Path(output).resolve(), Path(helper).resolve()
    bundle_data, bundle, workloads = load(root)
    selected = list(cases) if cases else list(workloads)
    if not selected or len(set(selected)) != len(selected) or set(selected)-set(workloads):
        raise ValueError("Select unique known cases")
    repository = Path(__file__).resolve().parents[3]
    # Only portable runtime sources must exist on the decoder host. Compiler
    # sources are pinned in the bundle but are neither imported nor executed.
    runtime_files = ("decoder_core/bitstream.py", "decoder_core/session.py", "decoder_core/decode.py",
                     "decoder_core/nvdec_decode.cpp", "codec_compute/workloads/temporal_benchmark.py",
                     "codec_compute/workloads/resident/benchmark.py")
    for name in runtime_files:
        if common.digest((repository/name).read_bytes()) != bundle["source_sha256"][name]:
            raise AssertionError("Runtime source differs from prepared bundle: " + name)
    Writer = common.writer_class(repository/"decoder_core/bitstream.py")
    for w in workloads.values():
        params = dict(w["case"]["writer"])
        params["reference_slots"] = {int(k): v for k, v in params["reference_slots"].items()}
        w["writer"] = Writer(**params)
    report = {"format": "codec-compute-resident-benchmark-1", "status": "running", "backend": "nvdec",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": bundle["source_commit"],
              "bundle_sha256": common.digest(bundle_data), "source_sha256": bundle["source_sha256"],
              "helper_sha256": common.digest(helper.read_bytes()), "repeats": repeats,
              "python": sys.version.split()[0],
              "gstreamer": subprocess.check_output(["gst-launch-1.0", "--version"], universal_newlines=True).splitlines()[0],
              "jobs_per_trial": bundle["jobs"], "timed_cases": selected, "cases": {},
              "protocol": {
                  "cold": "One new helper/pipeline per request; includes creation, input IPC, decode, selected I420 IPC, exact comparisons, EOS and final diagnostics",
                  "resident": "One new session per trial, one untimed complete warmup request, then sequential timed requests; each returns before next submission; creation/warmup/close separately recorded",
                  "host": "Includes input validation/serialization and byte-for-byte prefix check; appends cached prediction AUs and IDR fence",
                  "preencoded": "Uses preloaded frozen access units; input serialization excluded",
                  "fence": "One extra cached constant128 IDR per request; counted in bytes/pictures; previous fence discarded on next request, final fence drained at sole EOS",
                  "shared": "Same AU protocol, geometry, bytes, selections and exact full-returned-I420 checks in cold and resident modes; order rotates by repeat",
                  "excluded": "Compilation, fixture/reference generation, file loading, initial artifact/source audits, network and report writes",
                  "limits": "Fixed fields/shape, independent IDR jobs, fresh input uploads each request. No CUDA/concurrent baseline, resident camera references, or zero-copy output."}}

    def save():
        common.write_json(output, report)

    def session(w):
        return NVDECSession(width=w["case"]["coded_width"], height=w["case"]["coded_height"], helper=helper, timeout=30)

    def encode(w, job):
        prefix = tuple(w["writer"].input_luma(job["boards"][name], index)
                       for index, name in enumerate(w["case"]["input_names"]))
        if prefix != job["prefix"]:
            raise AssertionError("Runtime serialized input differs from frozen prefix")
        return prefix+w["tail"]

    def submit(active, w, job, *, host=False, full=False):
        units = encode(w, job) if host else job["units"]
        selection = None if full else tuple(sorted(set(w["case"]["output_frames"].values())))
        frames = active.submit(units, selected_frames=selection, fence=w["fence"])
        wanted = job["frames"] if full else tuple(job["frames"][i] for i in selection)
        if tuple(frames) != wanted:
            raise AssertionError("Actual resident decoded I420 differs")

    def check_counts(active, w, count, copied):
        stats = dict(active.stats)
        expected = {"requests": count, "pipeline_starts": 1,
                    "submitted_frames": count*(w["case"]["costs"]["total_pictures"]+1),
                    "decoded_frames": count*(w["case"]["costs"]["total_pictures"]+1),
                    "copied_frames": copied, "fence_pictures": count, "decoded_fences": count,
                    "pending_fences": 0, "fence_bytes": count*len(w["fence"])}
        if any(stats.get(name) != value for name, value in expected.items()):
            raise AssertionError("Final resident counters differ: " + repr(stats))
        return stats

    # Validate all fixtures and both plans, not only timed selections. This uses
    # the same live path without EOS between requests, and reads every real AU.
    for key, w in workloads.items():
        active = session(w)
        with active:
            for job in w["jobs"]:
                encode(w, job)
                submit(active, w, job, full=True)
        stats = check_counts(active, w, len(w["jobs"]), len(w["jobs"])*w["case"]["costs"]["total_pictures"])
        report["cases"][key] = {"costs": w["case"]["costs"], "level_idc": w["case"]["level_idc"],
                               "size": w["case"]["size"], "fence_bytes_per_request": len(w["fence"]),
                               "validation": {"status": "pass", **stats}, "modes": {}}
        print("PASS resident full validation " + key, flush=True)
        save()

    modes = ("input-serialization", "cold-preencoded", "resident-preencoded", "cold-host", "resident-host")
    entries = [(key, mode) for key in selected for mode in modes]
    for repeat in range(repeats):
        rotation = repeat*7 % len(entries)
        for key, mode in entries[rotation:]+entries[:rotation]:
            w = workloads[key]
            record = report["cases"][key]["modes"].setdefault(mode, {"trials": []})
            trial = {"request_ms": []}
            output_count = len(set(w["case"]["output_frames"].values()))
            if mode == "input-serialization":
                encode(w, w["jobs"][0])
                started = time.perf_counter()
                for job in w["jobs"]:
                    tick = time.perf_counter()
                    encode(w, job)
                    trial["request_ms"].append((time.perf_counter()-tick)*1000)
                trial["elapsed_ms"] = (time.perf_counter()-started)*1000
            elif mode.startswith("resident"):
                tick = time.perf_counter()
                active = session(w)
                trial["create_ms"] = (time.perf_counter()-tick)*1000
                try:
                    tick = time.perf_counter()
                    submit(active, w, w["jobs"][-1], host=mode.endswith("host"))
                    trial["warmup_ms"] = (time.perf_counter()-tick)*1000
                    started = time.perf_counter()
                    for job in w["jobs"]:
                        tick = time.perf_counter()
                        submit(active, w, job, host=mode.endswith("host"))
                        trial["request_ms"].append((time.perf_counter()-tick)*1000)
                    trial["elapsed_ms"] = (time.perf_counter()-started)*1000
                finally:
                    tick = time.perf_counter()
                    active.close()
                    trial["close_ms"] = (time.perf_counter()-tick)*1000
                trial["stats_including_warmup"] = check_counts(active, w, len(w["jobs"])+1,
                                                                (len(w["jobs"])+1)*output_count)
            else:
                # Warm the path but deliberately recreate the process/pipeline
                # for every measured request, including its final drain.
                with session(w) as warm:
                    submit(warm, w, w["jobs"][-1], host=mode.endswith("host"))
                started = time.perf_counter()
                trial["session_stats"] = []
                for job in w["jobs"]:
                    tick = time.perf_counter()
                    with session(w) as active:
                        submit(active, w, job, host=mode.endswith("host"))
                    trial["request_ms"].append((time.perf_counter()-tick)*1000)
                    trial["session_stats"].append(check_counts(active, w, 1, output_count))
                trial["elapsed_ms"] = (time.perf_counter()-started)*1000
            record["trials"].append(trial)
            record.update(summarize([item["elapsed_ms"] for item in record["trials"]], len(w["jobs"])))
            save()
        print("PASS timing repeat " + str(repeat+1), flush=True)
    report["status"] = "pass"
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    run(args.bundle, args.output, helper=args.helper, repeats=args.repeats, cases=args.cases)
