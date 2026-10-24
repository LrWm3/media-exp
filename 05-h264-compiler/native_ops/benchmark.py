#!/usr/bin/env python3
"""Measure host construction and complete decoder pipelines, including startup."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h264
from decoder_core.decode import available, decode
from common import BACKENDS, ROOT, digest, provenance, save_plane


def summary(samples):
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples}


def measure(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns()-start)/1e6)
    return summary(samples)


def kernels(steps):
    def pair(a, b):
        return h264.saturating_add(a, b)

    def resident(a, b):
        value = a
        for _ in range(steps):
            value = h264.saturating_add(value, b)
        return value
    return pair, resident


def inputs(width, height):
    return {"a": [[(3*x+5*y) % 32 for x in range(width)] for y in range(height)],
            "b": [[(x+2*y) % 3 for x in range(width)] for y in range(height)]}


def construct_costs(factory, payload, repeats):
    compiled = factory()
    cold_tail = []
    for _ in range(repeats):
        program = factory()  # Graph construction excluded from tail timing.
        start = time.perf_counter_ns()
        _ = program.tail
        cold_tail.append((time.perf_counter_ns()-start)/1e6)
    _ = compiled.tail
    return compiled, {
        "compile_graph_lower_schedule": measure(factory, repeats),
        "build_prediction_tail_once": summary(cold_tail),
        "read_cached_tail": measure(lambda: compiled.tail, 1000),
        "encode_input_pcm_and_validate": measure(lambda: compiled.encode_inputs(payload), repeats),
        "encode_inputs_plus_cached_tail": measure(lambda: compiled.encode(payload), repeats),
    }


def workload(program, payload, logical_operations):
    return {"logical_operations": logical_operations, "input_pictures": program.input_count,
            "prediction_pictures": len(program.operations), "refresh_copy_pictures": len(program.refreshes),
            "total_pictures": program.pictures,
            "pictures_per_logical_operation": program.pictures/logical_operations,
            "input_bytes": len(program.encode_inputs(payload)), "prediction_bytes": len(program.tail),
            "stream_bytes": len(program.encode(payload)),
            "readback_bytes": program.pictures*program.writer.width*program.writer.height*3//2,
            "output_frames": program.output_frames}


def report_markdown(report):
    config = report["configuration"]
    lines = ["# Native-operation benchmark", "",
             f"{config['width']} x {config['height']} active pixels; {config['steps']} saturating additions; "
             f"{config['repeats']} timed repetitions per decoder. All times are wall-clock medians.", "",
             "Hardware: " + "; ".join(report["provenance"]["display_devices"]) + ".", "",
             "The resident workload loads two inputs once and keeps the accumulator and increment image "
             "inside the decoded-picture reference buffer. Fresh pairs reload both operands for each addition. "
             "They are measured with either one decoder start per pair or one decoder session for all pairs. "
             "All three compute the same final pixels. Fresh operand fixtures are precomputed on the host; "
             "this fixed-workload benchmark does not model a host feedback loop.", "",
             "## Host construction", "",
             "| Phase | Pair kernel (ms) | Resident chain (ms) |", "| --- | ---: | ---: |"]
    labels = {"compile_graph_lower_schedule": "Compile graph, lower, schedule",
              "build_prediction_tail_once": "Build prediction bytes once",
              "read_cached_tail": "Read cached bytes",
              "encode_input_pcm_and_validate": "Encode/validate two PCM inputs",
              "encode_inputs_plus_cached_tail": "Encode inputs + concatenate cached tail"}
    for key, label in labels.items():
        lines.append(f"| {label} | {report['construction']['fresh_pair'][key]['median_ms']:.6f} | "
                     f"{report['construction']['resident'][key]['median_ms']:.6f} |")
    lines += ["", "## Input loading", "",
              "| Build complete input-bearing stream | Median (ms) |", "| --- | ---: |",
              f"| Encode all {config['steps']} fresh pairs + concatenate cached tails | "
              f"{report['input_loading']['fresh_pairs']['median_ms']:.3f} |",
              f"| Encode one resident pair + concatenate cached tail | "
              f"{report['input_loading']['resident']['median_ms']:.3f} |",
              "", "Operand fixtures already exist before these timers start. Both measurements include "
              "input validation, PCM encoding, and stream concatenation. Prediction tails are cached.",
              "", "## Complete workloads", "",
              "| Quantity | Fresh pairs, separate sessions | Fresh pairs, one session | Resident chain |",
              "| --- | ---: | ---: | ---: |"]
    for key, label in [("logical_operations", "Logical additions"), ("input_pictures", "PCM input pictures"),
                       ("prediction_pictures", "Prediction pictures"), ("refresh_copy_pictures", "Refresh copies"),
                       ("total_pictures", "All decoded pictures"), ("decoder_starts", "Decoder process starts"),
                       ("input_bytes", "Encoded input bytes"), ("prediction_bytes", "Prediction bytes"),
                       ("stream_bytes", "All stream bytes"), ("readback_bytes", "All YUV readback bytes")]:
        lines.append(f"| {label} | {report['workloads']['fresh_pairs'][key]:,} | "
                     f"{report['workloads']['fresh_pairs_one_session'][key]:,} | "
                     f"{report['workloads']['resident'][key]:,} |")
    lines += ["", "## Decode, readback, and output validation", "",
              "| Decoder | Fresh / separate sessions (ms) | Fresh / one session (ms) | Resident (ms) | Exact result |",
              "| --- | ---: | ---: | ---: | --- |"]
    for backend, result in report["decoders"].items():
        if result["status"] != "pass":
            lines.append(f"| {backend} | unavailable | unavailable | unavailable | — |")
        else:
            lines.append(f"| {backend} | {result['fresh_pairs']['median_ms']:.3f} | "
                         f"{result['fresh_pairs_one_session']['median_ms']:.3f} | "
                         f"{result['resident']['median_ms']:.3f} | pass |")
    lines += ["", "Each decoder measurement includes external process startup, parsing, decoding, "
              "all-picture YUV readback, and Python output/guard validation. Stream construction and "
              "prewriting benchmark files are excluded from these decode measurements and measured separately above. "
              "GStreamer backends also include writing and reading a temporary YUV file. This is an application "
              "pipeline measurement, not an isolated device-kernel benchmark.", "",
              "The separate-session comparison includes fewer decoder startups. The single-session comparison "
              "holds startup count at one while comparing repeated input loads/IDR resets against resident references. "
              "Both comparisons include fewer PCM pictures and less readback in the resident case. "
              "They do not establish GPU arithmetic throughput or superiority over CPU/vectorized "
              "addition. No NVIDIA hardware or NVDEC was tested.", "",
              "Prediction bytes are input-independent and reused without rebuilding. Reference refresh copies "
              "extend live values beyond the 16-picture DPB window; they consume both pictures and stream bytes. "
              "The workload report counts these copies explicitly. The runtime still loads inputs using ordinary "
              "H.264 PCM pictures; it cannot directly inject host buffers into decoder references.", "",
              "See `benchmark.json` for every timing sample, stream size, output hash, and machine provenance.", ""]
    return "\n".join(lines)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    data = inputs(args.width, args.height)
    pair_kernel, resident_kernel = kernels(args.steps)
    options = {"width": args.width, "height": args.height,
               "input_ranges": {"a": (0, 255), "b": (0, 255)}}
    pair, pair_cost = construct_costs(lambda: h264.compile(pair_kernel, **options), data, args.repeats)
    resident, resident_cost = construct_costs(lambda: h264.compile(resident_kernel, **options), data, args.repeats)
    expected = [[min(255, a+args.steps*b) for a, b in zip(ar, br)]
                for ar, br in zip(data["a"], data["b"])]
    if resident.reference(data)["output"] != expected:
        raise AssertionError("Resident graph differs from repeated saturating addition")
    resident.save(args.output/"streams"/"benchmark-resident.h264", data)
    pair.save(args.output/"streams"/"benchmark-fresh-pair.h264", data)
    resident_work = workload(resident, data, args.steps)
    resident_work["decoder_starts"] = 1
    fresh_work = {key: 0 for key in ("logical_operations", "input_pictures", "prediction_pictures",
                                   "refresh_copy_pictures", "total_pictures", "input_bytes", "prediction_bytes",
                                   "stream_bytes", "readback_bytes")}
    fresh_work["decoder_starts"] = args.steps
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance(),
              "configuration": {"width": args.width, "height": args.height, "steps": args.steps,
                                "repeats": args.repeats, "timing": "wall clock; decoder startup included"},
              "construction": {"fresh_pair": pair_cost, "resident": resident_cost},
              "workloads": {"fresh_pairs": fresh_work, "resident": resident_work}, "decoders": {}}
    with tempfile.TemporaryDirectory(prefix="native-ops-benchmark-") as temporary:
        temporary = Path(temporary)
        resident_path = temporary/"resident.h264"
        resident_path.write_bytes(resident.encode(data))
        paths, expected_steps, payloads = [], [], []
        current = data["a"]
        for index in range(args.steps):
            payload = {"a": current, "b": data["b"]}
            payloads.append(payload)
            path = temporary/f"pair-{index:03d}.h264"
            path.write_bytes(pair.encode(payload))
            paths.append(path)
            counts = workload(pair, payload, 1)
            for key in fresh_work:
                if key != "decoder_starts":
                    fresh_work[key] += counts[key]
            current = [[min(255, a+b) for a, b in zip(ar, br)]
                       for ar, br in zip(current, data["b"])]
            expected_steps.append(current)
        fresh_work["pictures_per_logical_operation"] = fresh_work["total_pictures"]/args.steps
        report["workloads"]["fresh_pairs_one_session"] = {**fresh_work, "decoder_starts": 1}
        report["input_loading"] = {
            "fresh_pairs": measure(lambda: b"".join(pair.encode(payload) for payload in payloads), args.repeats),
            "resident": measure(lambda: resident.encode(data), args.repeats),
            "fixtures_precomputed_on_host": True, "host_feedback_loop_modeled": False,
        }
        joined_path = temporary/"fresh-one-session.h264"
        joined_path.write_bytes(b"".join(path.read_bytes() for path in paths))
        (args.output/"streams"/"benchmark-fresh-one-session.h264").write_bytes(joined_path.read_bytes())
        for backend in args.decoders:
            if not available(backend):
                report["decoders"][backend] = {"status": "unavailable"}
                continue
            def fresh_pipeline():
                for path, expected_step in zip(paths, expected_steps):
                    actual = pair.read(decode(path, backend))["output"]
                    if actual != expected_step:
                        raise AssertionError(f"{backend}: fresh-pair result differs")
            def resident_pipeline():
                actual = resident.read(decode(resident_path, backend))["output"]
                if actual != expected:
                    raise AssertionError(f"{backend}: resident result differs")
                return actual
            def one_session_pipeline():
                raw = decode(joined_path, backend)
                chunk_bytes = pair.pictures*pair.writer.width*pair.writer.height*3//2
                if len(raw) != chunk_bytes*args.steps:
                    raise AssertionError(f"{backend}: concatenated stream picture count differs")
                for index, expected_step in enumerate(expected_steps):
                    actual = pair.read(raw[index*chunk_bytes:(index+1)*chunk_bytes])["output"]
                    if actual != expected_step:
                        raise AssertionError(f"{backend}: one-session fresh result differs")
            # Same sequence per backend, no untimed decoder warmup. Repeated samples
            # expose startup/cache variation rather than hiding it in a kernel rate.
            fresh_times = measure(fresh_pipeline, args.repeats)
            one_session_times = measure(one_session_pipeline, args.repeats)
            resident_times = measure(resident_pipeline, args.repeats)
            actual = resident_pipeline()
            save_plane(actual, args.output/"captures"/f"benchmark-{backend}-output.png")
            report["decoders"][backend] = {"status": "pass", "fresh_pairs": fresh_times,
                                           "fresh_pairs_one_session": one_session_times,
                                           "resident": resident_times, "output_sha256": digest(actual)}
            print(f"PASS {backend}: fresh {fresh_times['median_ms']:.1f} ms; "
                  f"fresh one-session {one_session_times['median_ms']:.1f} ms; "
                  f"resident {resident_times['median_ms']:.1f} ms", flush=True)
    (args.output/"benchmark.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"BENCHMARK.md").write_text(report_markdown(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--decoders", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
