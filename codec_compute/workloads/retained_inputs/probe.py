"""Observe actual retained-stream output at paused access-unit boundaries."""

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import time

spec = importlib.util.spec_from_file_location("_retained_probe_runtime", Path(__file__).with_name("runtime.py"))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def run(root, output, *, nvdec_helper, wait_ms=50):
    if type(wait_ms) is not int or not 1 <= wait_ms <= 1000:
        raise ValueError("Probe wait must be 1..1000 milliseconds")
    data, bundle, workloads, groups = runtime.load(root)
    helper = Path(nvdec_helper).resolve()
    report = dict(format="codec-compute-retained-availability-1", status="running",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  source_commit=bundle["source_commit"], source_sha256=bundle["source_sha256"],
                  bundle_sha256=runtime.common.digest(data),
                  helper_sha256=runtime.common.digest(helper.read_bytes()), wait_ms=wait_ms, settle_ms=500, cases={},
                  contract="One AU appended, then bounded collection through that AU. Append is not completion. Every observed I420 frame checked. Sole final EOS returns remaining pictures. No fence, DPB-disable option, extra lookahead picture, surface import or consumer.",
                  scope="Availability diagnostic with artificial bounded polling; not a throughput benchmark or proof that timed-out hardware execution did not finish internally.")

    def save():
        runtime.common.write_json(output, report)

    try:
        for key, w in workloads.items():
            case, records = w["case"], []
            report["cases"][key] = records
            for index, job in enumerate(w["jobs"]):
                active = runtime.NVDECStream(case["coded_width"], case["coded_height"], helper=helper, timeout=20)
                record = dict(fixture=index, steps=[], status="running")
                records.append(record)
                checked = []

                def verify(frames):
                    for frame, pixels in frames:
                        if frame != len(checked) or pixels != job["frames"][frame]:
                            raise AssertionError("Continuous NVDEC output differs or is missing")
                        checked.append(frame)

                try:
                    for frame, unit in enumerate(job["units"]):
                        before = active.stats
                        tick = time.perf_counter()
                        active.append((unit,))
                        enqueue_ms = (time.perf_counter()-tick)*1000
                        after_append = active.stats
                        if after_append["decoded_frames"] != before["decoded_frames"]:
                            raise AssertionError("Append falsely claims decode completion")
                        tick = time.perf_counter()
                        actual = active.collect(frame+1, range(before["decoded_frames"], frame+1), wait_ms=wait_ms)
                        elapsed = (time.perf_counter()-tick)*1000
                        verify(actual)
                        stats = active.stats
                        record["steps"].append(dict(submitted_frame=frame,
                            input_picture=frame < len(case["input_names"]),
                            append_ms=enqueue_ms, collect_ms=elapsed,
                            returned_indices=[i for i,_ in actual],
                            target_complete=stats["decoded_frames"] == frame+1, stats=stats))
                    record["before_settle"] = active.stats
                    tick = time.perf_counter()
                    verify(active.collect(len(job["units"]),
                                          range(active.stats["decoded_frames"], len(job["units"])), wait_ms=500))
                    record["settle_ms"] = (time.perf_counter()-tick)*1000
                    record["before_eos"] = active.stats
                    tick = time.perf_counter()
                    verify(active.close(range(active.stats["decoded_frames"], len(job["units"]))))
                    record["close_ms"] = (time.perf_counter()-tick)*1000
                finally:
                    active.close()
                stats, total = active.stats, len(job["units"])
                expected = dict(requests=2*total+2, submitted_frames=total, decoded_frames=total,
                                copied_frames=total, pending_frames=0, pipeline_starts=1,
                                input_bytes=sum(map(len,job["units"])), outstanding_input_bytes=0)
                if stats != expected or checked != list(range(total)):
                    raise AssertionError("Final continuous stream accounting differs")
                record.update(status="pass", after_close=stats,
                              output_boundaries_complete=sum(s["target_complete"] for s in record["steps"]
                                                             if not s["input_picture"]),
                              output_boundaries=case["output_count"])
                save()
            print("PASS retained availability "+key, flush=True)
        for name, checksum in bundle["source_sha256"].items():
            if runtime.common.digest((runtime.REPOSITORY/name).read_bytes()) != checksum:
                raise AssertionError("Source changed during probe")
        if (runtime.common.digest((Path(root)/"bundle.json").read_bytes()) != report["bundle_sha256"]
                or runtime.common.digest(helper.read_bytes()) != report["helper_sha256"]):
            raise AssertionError("Probe input changed")
        report["status"] = "pass"
        save()
    except Exception as error:
        report.update(status="failed", error=dict(type=type(error).__name__, detail=str(error)))
        save()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--nvdec-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-ms", type=int, default=50)
    args = parser.parse_args()
    run(args.bundle, args.output, nvdec_helper=args.nvdec_helper, wait_ms=args.wait_ms)
