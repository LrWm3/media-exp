#!/usr/bin/env python3
"""Fixed-window foreground deadlines and background progress on one logical CPU.

Python 3.8+, standard library only. Run this file directly on the measurement
host; importing the semantic compiler is unnecessary. No hardware or energy
advantage is inferred automatically from these implementation measurements.
"""

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import hmac
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import time


HERE = Path(__file__).resolve().parent
MODES = ("background-only", "cpu", "nvdec-baseline", "nvdec-merged")
GROUP = "512/resident-chain-64"
PASSWORD = b"codec-compute-public-background-work"
SALT = b"fixed-window-public-salt-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(path)


def wait_until(target, clock=time.monotonic, sleep=time.sleep):
    while True:
        remaining = target-clock()
        if remaining <= 0:
            return
        sleep(min(remaining, 0.05))


def thread_masks(pid):
    result = {}
    for path in Path("/proc/{}/task".format(pid)).iterdir():
        try:
            result[int(path.name)] = sorted(os.sched_getaffinity(int(path.name)))
        except ProcessLookupError:
            pass
    if not result:
        raise RuntimeError("No live process threads to audit")
    return result


def audit_affinity(pid, logical_cpu):
    masks = thread_masks(pid)
    if any(mask != [logical_cpu] for mask in masks.values()):
        raise AssertionError("A measured process thread escaped the one-CPU affinity budget")
    priorities, policies = set(), set()
    for tid in masks:
        try:
            priorities.add(os.getpriority(os.PRIO_PROCESS, tid))
            policies.add(os.sched_getscheduler(tid))
        except ProcessLookupError:
            pass
    nice, policy = os.getpriority(os.PRIO_PROCESS, pid), os.sched_getscheduler(pid)
    if priorities != {nice} or policies != {policy}:
        raise AssertionError("Process threads have unequal scheduling priority or policy")
    return {"threads": len(masks), "allowed_logical_cpus": [logical_cpu],
            "nice": nice, "scheduler_policy": policy,
            "thread_nice_values": sorted(priorities), "thread_scheduler_policies": sorted(policies)}


def equal_priorities(audits):
    reference = next(iter(audits.values()))
    if any(audit[field] != reference[field] for audit in audits.values() for field in ("nice", "scheduler_policy")):
        raise AssertionError("Measured processes have unequal scheduling priority or policy")


@contextmanager
def one_cpu_budget():
    original = thread_masks(os.getpid())
    original_main = sorted(os.sched_getaffinity(0))
    logical_cpu = min(original_main)
    try:
        for tid in original:
            try:
                os.sched_setaffinity(tid, {logical_cpu})
            except ProcessLookupError:
                pass
        yield logical_cpu
    finally:
        # Restore existing threads individually; any new runner threads inherit
        # the runner's previous allowed set. No other process/global mask changes.
        for tid in thread_masks(os.getpid()):
            try:
                os.sched_setaffinity(tid, original.get(tid, original_main))
            except ProcessLookupError:
                pass


def chunk(iterations):
    return hashlib.pbkdf2_hmac("sha256", PASSWORD, SALT, iterations, dklen=32)


def calibrate(target_seconds=0.001):
    iterations, measurements = 2048, []
    for _ in range(3):
        samples = []
        for _ in range(5):
            start = time.monotonic()
            chunk(iterations)
            samples.append(time.monotonic()-start)
        measurements.append({"iterations": iterations, "wall_ms": [value*1000 for value in samples]})
        adjusted = max(1, min(1000000, round(iterations*target_seconds/statistics.median(samples))))
        if len(measurements) < 3:
            iterations = adjusted
    return {"algorithm": "PBKDF2-HMAC-SHA256", "password_ascii": PASSWORD.decode(), "salt_ascii": SALT.decode(),
            "derived_key_bytes": 32, "iterations": iterations, "expected_hex": chunk(iterations).hex(),
            "target_chunk_ms": target_seconds*1000, "calibration": measurements}


def _background_worker(connection, logical_cpu, specification):
    try:
        os.sched_setaffinity(0, {logical_cpu})
        iterations = specification["iterations"]
        expected = bytes.fromhex(specification["expected_hex"])
        if not hmac.compare_digest(chunk(iterations), expected):
            raise AssertionError("Background warmup checksum differs")
        connection.send({"status": "ready", "affinity": audit_affinity(os.getpid(), logical_cpu)})
        if not connection.poll(30):
            raise TimeoutError("Background start barrier was not released")
        command = connection.recv()
        start, end = command["start"], command["end"]
        connection.send({"status": "armed"})
        wait_until(start)
        observed_start = time.monotonic()
        cpu_start = time.process_time()
        completed, checked, boundary = 0, 0, None
        last_completion = None
        while time.monotonic() < end:
            before = time.monotonic()
            chunk_cpu = time.process_time()
            if not hmac.compare_digest(chunk(iterations), expected):
                raise AssertionError("Background chunk checksum differs")
            finished = time.monotonic()
            used_cpu = time.process_time()-chunk_cpu
            checked += 1
            if finished <= end:
                completed += 1
                last_completion = finished-start
            else:
                boundary = {"start_offset_ms": (before-start)*1000, "finish_offset_ms": (finished-start)*1000,
                            "cpu_ms": used_cpu*1000, "overrun_ms": (finished-end)*1000}
        cpu_end, observed_end = time.process_time(), time.monotonic()
        connection.send({"status": "done", "completed_chunks": completed, "checked_chunks": checked,
                         "chunks_per_second": completed/(end-start), "last_completion_offset_s": last_completion,
                         "start_lateness_ms": max(0, observed_start-start)*1000,
                         "finish_overrun_ms": max(0, observed_end-end)*1000,
                         "work_process_cpu_ms": (cpu_end-cpu_start)*1000, "boundary_chunk": boundary,
                         "affinity": audit_affinity(os.getpid(), logical_cpu)})
        # Remain alive for the parent's live process CPU snapshot. Foreground
        # overload may finish after the fixed background window has ended.
        while not connection.poll(1):
            pass
        connection.recv()
    except BaseException as error:
        try:
            connection.send({"status": "error", "error": type(error).__name__+": "+str(error)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Background:
    def __init__(self, logical_cpu, specification):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_background_worker, args=(child, logical_cpu, specification), daemon=True)
        self.process.start()
        child.close()
        try:
            self.ready = self.receive("ready", 30)
        except BaseException:
            self.close()
            raise

    def receive(self, status, timeout=10):
        if not self.connection.poll(min(60, timeout)):
            raise TimeoutError("Background worker did not report "+status)
        result = self.connection.recv()
        if result.get("status") != status:
            raise RuntimeError("Background worker: "+repr(result))
        return result

    def arm(self, start, end):
        self.connection.send({"start": start, "end": end})
        self.receive("armed", 10)
        if time.monotonic() >= start:
            raise RuntimeError("Background start barrier was acknowledged too late")

    def close(self):
        try:
            if self.process.is_alive():
                self.connection.send({"status": "close"})
                self.process.join(1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
            else:
                self.process.join(1)
        except (BrokenPipeError, EOFError, OSError):
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        finally:
            self.connection.close()


def releases(operation, count, start, period, deadline, clock=time.monotonic, sleep=time.sleep):
    jobs = []
    for index in range(count):
        release = start+index*period
        wait_until(release, clock, sleep)
        began = clock()
        operation(index)
        finished = clock()
        jobs.append({"index": index, "fixture": index % 8, "release_offset_s": index*period,
                     "start_offset_s": began-start, "finish_offset_s": finished-start,
                     "release_lateness_ms": max(0, began-release)*1000,
                     "service_ms": (finished-began)*1000, "release_to_completion_ms": (finished-release)*1000,
                     "deadline_miss": finished > release+deadline})
    return jobs


def trial(corpus, mode, *, logical_cpu, background_spec, accounting_factory,
          duration=12.0, period=1.0, deadline=0.75, counter_check=None):
    if mode not in MODES:
        raise ValueError("Unknown concurrency mode")
    if duration <= 0 or period <= 0 or deadline <= 0:
        raise ValueError("Window, release period and deadline must be positive")
    count = round(duration/period)
    if count < 1 or not math.isclose(count*period, duration):
        raise ValueError("Window must contain a positive whole number of release periods")
    with ExitStack() as stack:
        active, key, worker_pid = None, GROUP+"/baseline", None
        setup_start = time.monotonic()
        if mode.startswith("nvdec-"):
            if counter_check is None:
                raise ValueError("NVDEC trials require a final counter checker")
            key = GROUP+"/"+mode.split("-", 1)[1]
            active = stack.enter_context(corpus.session(key))
            worker_pid = active._process.pid
        setup_ms = (time.monotonic()-setup_start)*1000
        warmup_start = time.monotonic()
        if mode == "cpu":
            corpus.cpu(key, 7)
        elif active is not None:
            corpus.submit(active, key, 7)
        warmup_ms = (time.monotonic()-warmup_start)*1000
        foreground = accounting_factory(worker_pid)
        background = Background(logical_cpu, background_spec)
        stack.callback(background.close)
        background_clock = accounting_factory(background.process.pid)
        masks_before = {"runner": audit_affinity(os.getpid(), logical_cpu), "background": background.ready["affinity"]}
        if worker_pid is not None:
            masks_before["decoder"] = audit_affinity(worker_pid, logical_cpu)
        equal_priorities(masks_before)
        start = time.monotonic()+0.25
        end = start+duration
        background.arm(start, end)
        wait_until(start)
        accounting_start = time.monotonic()
        fg_start, bg_start = foreground.snapshot(), background_clock.snapshot()
        if mode == "background-only":
            jobs = []
        else:
            operation = ((lambda index: corpus.cpu(key, index)) if mode == "cpu" else
                         (lambda index: corpus.submit(active, key, index)))
            jobs = releases(operation, count, start, period, deadline)
        foreground_finished = start+(jobs[-1]["finish_offset_s"] if jobs else 0)
        wait_until(end)
        progress = background.receive("done", 10)
        fg_end, bg_end = foreground.snapshot(), background_clock.snapshot()
        accounting_end = time.monotonic()
        delta = {name: fg_end[name]-value for name, value in fg_start.items()}
        bg_cpu = bg_end["worker_cpu_s"]-bg_start["worker_cpu_s"]
        if min(list(delta.values())+[bg_cpu]) < -1e-6:
            raise RuntimeError("Process CPU clock went backwards")
        foreground_cpu_ms = 1000*(delta["parent_cpu_s"]+delta.get("worker_cpu_s", 0))
        masks_after = {"runner": audit_affinity(os.getpid(), logical_cpu), "background": progress["affinity"]}
        if worker_pid is not None:
            masks_after["decoder"] = audit_affinity(worker_pid, logical_cpu)
        equal_priorities(masks_after)
        result = {"mode": mode, "window_seconds": duration, "scheduled_jobs": 0 if mode == "background-only" else count,
                "completed_jobs": len(jobs), "deadline_misses": sum(job["deadline_miss"] for job in jobs),
                "foreground_window_overrun_ms": max(0, foreground_finished-end)*1000,
                "foreground_jobs": jobs, "background": progress,
                "foreground_cpu_deltas_s": delta, "foreground_process_cpu_ms": foreground_cpu_ms,
                "background_live_process_cpu_ms": bg_cpu*1000,
                "combined_process_cpu_ms": foreground_cpu_ms+bg_cpu*1000,
                "accounting_start_lateness_ms": max(0, accounting_start-start)*1000,
                "accounting_end_overrun_ms": max(0, accounting_end-end)*1000,
                "affinity_before": masks_before, "affinity_after": masks_after,
                "excluded_setup_ms": setup_ms, "excluded_warmup_ms": warmup_ms,
                "decoder_stats_before_close": dict(active.stats) if active is not None else None}
    if active is not None:
        case = corpus.workloads[key]["case"]
        requests = len(jobs)+1
        copies = requests*len(set(case["output_frames"].values()))
        result["decoder_stats_after_close"] = dict(counter_check(active, case, requests, copies))
        result["decoder_counter_scope"] = "Includes one warmup request; final close and counter check are outside timing."
    return result


def rotation(repeat):
    offset = repeat % len(MODES)
    return MODES[offset:]+MODES[:offset]


def distribution(values):
    return {"samples": values, "median": statistics.median(values), "range": [min(values), max(values)]}


def summarize(trials):
    result = {}
    controls = {row["repeat"]: row["background"]["chunks_per_second"] for row in trials if row["mode"] == "background-only"}
    cpu = {row["repeat"]: row["background"]["chunks_per_second"] for row in trials if row["mode"] == "cpu"}
    for mode in MODES:
        rows = [row for row in trials if row["mode"] == mode]
        rates = [row["background"]["chunks_per_second"] for row in rows]
        result[mode] = {"background_chunks_per_second": distribution(rates),
                        "deadline_misses": sum(row["deadline_misses"] for row in rows),
                        "foreground_jobs": sum(row["completed_jobs"] for row in rows),
                        "foreground_cpu_ms": distribution([row["foreground_process_cpu_ms"] for row in rows]),
                        "background_cpu_ms": distribution([row["background_live_process_cpu_ms"] for row in rows]),
                        "combined_cpu_ms": distribution([row["combined_process_cpu_ms"] for row in rows]),
                        "background_rate_difference_vs_cpu": distribution([rate-cpu[row["repeat"]] for rate, row in zip(rates, rows)])}
        result[mode]["deadline_status"] = ("not-applicable" if mode == "background-only" else
                                             "missed" if result[mode]["deadline_misses"] else "met")
        if all(controls[row["repeat"]] > 0 for row in rows):
            result[mode]["background_fraction_of_control"] = distribution([rate/controls[row["repeat"]] for rate, row in zip(rates, rows)])
        latencies = [job["release_to_completion_ms"] for row in rows for job in row["foreground_jobs"]]
        if latencies:
            result[mode]["release_to_completion_ms"] = distribution(latencies)
    return result


def run(root, output, *, cpu_helper, pcm_helper, nvdec_helper, build_info, repeats=5):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    spec = importlib.util.spec_from_file_location("_guard_concurrency_runtime", HERE/"runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    script_hash = digest(Path(__file__).read_bytes())
    report = {"format": "codec-compute-guard-concurrency-1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": script_hash,
              "group": GROUP, "repeats": repeats, "window_seconds": 12, "release_period_seconds": 1,
              "deadline_seconds": 0.75, "trials": [], "protocol": {
                  "affinity": "One first-allowed logical CPU shared by runner, decoder threads and one spawned background worker; masks verified; runner masks restored.",
                  "releases": "Twelve absolute 1 Hz releases per foreground trial, sequential execution without drops; deadline measured from release including queueing.",
                  "background": "Same public PBKDF2 input and iteration count in every mode; checksum checked each chunk; count only completions within the fixed 12-second window.",
                  "barrier": "Worker ready after setup/warmup, then armed acknowledgement and common future monotonic start/end timestamps.",
                  "overload": "All scheduled foreground jobs finish, even after window end; background stops independently at fixed end; overruns and boundary chunks remain explicit.",
                  "cpu_scope": "Live parent, decoder worker and background process CPU clocks. Accounting extends through foreground backlog and final boundary chunk; offsets recorded separately.",
                  "excluded": "Initialization, warmup, process creation and final close excluded. Required foreground work and output checks included. No clocks, priorities or global settings changed.",
                  "limits": "No energy inference. Accounting covers named processes; other processes and spawn bookkeeping are excluded. Driver IRQs/kernel workqueues are not constrained by process affinity. Raw ranges retained; small/noisy or null effects are valid outcomes.",
                  "status": "Pass means protocol, outputs and provenance verified. Deadline qualification is reported separately; throughput gains with missed deadlines do not establish the requested service level."}}
    try:
        with one_cpu_budget() as logical_cpu:
            report["logical_cpu"] = logical_cpu
            report["background_specification"] = calibrate()
            corpus = runtime.Corpus(root, cpu_helper=cpu_helper, pcm_helper=pcm_helper,
                                    nvdec_helper=nvdec_helper, build_info=build_info)
            try:
                report["identity"] = corpus.identity
                source_name = str(Path(__file__).resolve().relative_to(HERE.parents[2]))
                if corpus.bundle["source_sha256"].get(source_name) != script_hash:
                    raise AssertionError("Concurrency source differs from frozen bundle")
                validation = {"cpu_jobs": 0, "nvdec_jobs": {}, "full_i420": True, "affinity": {},
                              "decoder_stats_after_close": {}}
                for index in range(8):
                    corpus.cpu(GROUP+"/baseline", index)
                    validation["cpu_jobs"] += 1
                for variant in ("baseline", "merged"):
                    key = GROUP+"/"+variant
                    with corpus.session(key) as active:
                        for index in range(8):
                            corpus.submit(active, key, index, full=True)
                        validation["nvdec_jobs"][variant] = 8
                        validation["affinity"][variant] = audit_affinity(active._process.pid, logical_cpu)
                        equal_priorities({"runner": audit_affinity(os.getpid(), logical_cpu),
                                          "decoder": validation["affinity"][variant]})
                    case = corpus.workloads[key]["case"]
                    validation["decoder_stats_after_close"][variant] = dict(
                        runtime.counts(active, case, 8, 8*case["costs"]["total_pictures"]))
                report["validation"] = {"status": "pass", **validation}
                save(output, report)
                for repeat in range(repeats):
                    for order, mode in enumerate(rotation(repeat)):
                        row = trial(corpus, mode, logical_cpu=logical_cpu,
                                    background_spec=report["background_specification"], accounting_factory=runtime.Accounting,
                                    counter_check=runtime.counts)
                        row.update(repeat=repeat, order=order)
                        report["trials"].append(row)
                        save(output, report)
                        print("PASS repeat={} mode={} background={:.3f}/s misses={}".format(
                            repeat, mode, row["background"]["chunks_per_second"], row["deadline_misses"]), flush=True)
                corpus.verify_stable()
                if digest(Path(__file__).read_bytes()) != script_hash:
                    raise RuntimeError("Concurrency script changed during measurement")
                report["summary"] = summarize(report["trials"])
                report["deadline_qualification"] = {mode: {"status": data["deadline_status"],
                                                            "misses": data["deadline_misses"],
                                                            "jobs": data["foreground_jobs"]}
                                                    for mode, data in report["summary"].items()}
                report["status"] = "pass"
            finally:
                corpus.close()
        report["runner_affinity_restored"] = True
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = type(error).__name__+": "+str(error)
        save(output, report)
        raise
    save(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-helper", type=Path, required=True)
    parser.add_argument("--pcm-helper", type=Path, required=True)
    parser.add_argument("--nvdec-helper", type=Path, required=True)
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.bundle, args.output, cpu_helper=args.cpu_helper, pcm_helper=args.pcm_helper,
        nvdec_helper=args.nvdec_helper, build_info=args.build_info, repeats=args.repeats)
