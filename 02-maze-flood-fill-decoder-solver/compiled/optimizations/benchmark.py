#!/usr/bin/env python3
"""Validate paired maze bundles and time four modes using Python 3.8 + decoder_core."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

for parent in Path(__file__).resolve().parents:
    if (parent / "decoder_core" / "decode.py").exists():
        sys.path.insert(0, str(parent))
        break
from decoder_core.decode import _nvdec_helper, available, decode_batch


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_file(root, relative, checksum):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Bundle path escapes directory")
    data = path.read_bytes()
    if digest(data) != checksum:
        raise ValueError("Bundle hash mismatch: " + relative)
    return data


def active_plane(frame, case):
    width, height, stride, guard = (case[key] for key in ("width", "height", "coded_width", "guard"))
    return b"".join(frame[(y + guard) * stride + guard:(y + guard) * stride + guard + width]
                    for y in range(height))


def load_bundle(root):
    bundle_bytes = (root / "bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle.get("format") != "h264-maze-optimization-bundle-0.1":
        raise ValueError("Unsupported bundle format")
    if not bundle["cases"] or len(bundle["cases"]) != bundle["case_count"]:
        raise ValueError("Invalid bundle case count")
    loaded = {}
    for case in bundle["cases"]:
        name = case["name"]
        if name in loaded:
            raise ValueError("Duplicate case: " + name)
        named = {label: checked_file(root, item["path"], item["sha256"])
                 for label, item in case["expected_named_outputs"].items()}
        loaded[name] = {}
        for variant in ("baseline", "optimized"):
            item = case["variants"][variant]
            checked_file(root, item["stream"], item["stream_sha256"])
            manifest = json.loads(checked_file(root, item["manifest"], item["manifest_sha256"]))
            full = checked_file(root, item["expected"], item["expected_sha256"])
            frame_bytes = case["frame_bytes"]
            if frame_bytes != case["coded_width"] * case["coded_height"] * 3 // 2:
                raise ValueError("Invalid packed I420 geometry")
            if len(full) != item["pictures"] * frame_bytes or len(full) != item["expected_bytes"]:
                raise ValueError("Full reference picture count mismatch")
            selection = item["selected_frames"]
            if selection != sorted(set(item["output_frames"].values())):
                raise ValueError("Selection must cover exactly all named output pictures")
            if any(type(index) is not int or index < 0 or index >= item["pictures"] for index in selection):
                raise ValueError("Selected picture index out of range")
            if manifest["output_frames"] != item["output_frames"] or manifest["pictures"] != item["pictures"]:
                raise ValueError("Manifest differs from bundle picture mapping")
            if set(named) != set(item["output_frames"]) or set(named) != set(item["named_outputs"]):
                raise ValueError("Named output coverage mismatch")
            selected = b"".join(full[index * frame_bytes:(index + 1) * frame_bytes] for index in selection)
            if digest(selected) != item["selected_sha256"] or len(selected) != item["selected_bytes"]:
                raise ValueError("Selected reference hash mismatch")
            for label, frame in item["output_frames"].items():
                info = item["named_outputs"][label]
                if info["frame"] != frame or selection[info["selected_position"]] != frame:
                    raise ValueError("Named output position mismatch")
                values = active_plane(full[frame * frame_bytes:(frame + 1) * frame_bytes], case)
                if values != named[label] or digest(values) != info["sha256"]:
                    raise ValueError("Named reference mismatch: %s/%s/%s" % (name, variant, label))
            loaded[name][variant] = {"stream": root / item["stream"], "full": full,
                                     "selected": selected, "named": named, "case": case, "item": item}
    return bundle_bytes, bundle, loaded


def execute(entry, backend, selective):
    """Always use one decode_batch job, including full readback modes."""
    case, item = entry["case"], entry["item"]
    selection = item["selected_frames"] if selective else None
    result = decode_batch([entry["stream"]], backend,
                          width=case["coded_width"], height=case["coded_height"],
                          pictures_per_job=item["pictures"], selected_frames=selection)
    actual = b"".join(result.frames[0])
    if actual != entry["selected" if selective else "full"]:
        raise AssertionError("Full I420 byte comparison failed")
    expected_copies = len(selection) if backend == "nvdec" and selective else item["pictures"]
    if result.decoded_frames != item["pictures"] or result.copied_frames != expected_copies:
        raise AssertionError("Actual decoded/copied frame counts differ from readback contract")
    counters = {"decoder_sessions": 1, "decoded_frames": result.decoded_frames,
                "copied_frames": result.copied_frames, "returned_frames": len(result.frames[0]),
                "returned_bytes": len(actual), "copied_bytes": result.copied_frames * case["frame_bytes"],
                "counter_source": "validated helper diagnostics" if backend == "nvdec" else "full compatibility readback",
                "full_i420_byte_equality": True}
    return result, actual, counters


def check_named(result, entry):
    positions = {frame: index for index, frame in enumerate(result.selected_frames)}
    named = {}
    for label, frame in entry["item"]["output_frames"].items():
        values = active_plane(result.frames[0][positions[frame]], entry["case"])
        if values != entry["named"][label]:
            raise AssertionError("Actual named output differs: " + label)
        named[label] = values
    return named


def save_capture(output, result, actual, entry, backend, bundle_sha256):
    directory = output / "captures"
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / "demo-optimized-selected.i420"
    raw_path.write_bytes(actual)
    case = entry["case"]
    capture = {"backend": backend, "bundle_sha256": bundle_sha256,
               "stream_sha256": entry["item"]["stream_sha256"],
               "path": str(raw_path.relative_to(output)), "sha256": digest(actual), "bytes": len(actual),
               "selected_frames": list(result.selected_frames), "output_frames": entry["item"]["output_frames"],
               "width": case["width"], "height": case["height"], "coded_width": case["coded_width"],
               "coded_height": case["coded_height"], "guard": case["guard"], "frame_bytes": case["frame_bytes"],
               "decoded_frames": result.decoded_frames, "copied_frames": result.copied_frames,
               "named_outputs": {}}
    for label, values in check_named(result, entry).items():
        path = directory / "demo-optimized" / (label + ".pgm")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("P5\n%d %d\n255\n" % (case["width"], case["height"])).encode() + values)
        capture["named_outputs"][label] = {"path": str(path.relative_to(output)),
                                           "sha256": digest(path.read_bytes()), "plane_sha256": digest(values)}
    metadata = directory / "demo-optimized-selected.json"
    metadata.write_text(json.dumps(capture, indent=2) + "\n")
    return {**capture, "metadata": str(metadata.relative_to(output)), "metadata_sha256": digest(metadata.read_bytes())}


def benchmark(root, output, backend="nvdec", helper=None, repeats=5):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if helper is not None or backend == "nvdec":
        helper = Path(helper).resolve() if helper is not None else _nvdec_helper()
        os.environ["H264_NVDEC_DECODER"] = str(helper)
    if not available(backend):
        raise RuntimeError("Requested decoder unavailable: " + backend)
    bundle_bytes, bundle, loaded = load_bundle(root)
    if "demo" not in loaded:
        raise ValueError("Bundle needs demo case for the paired four-mode benchmark")
    report = {"format": "h264-maze-optimization-results-0.1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "backend": backend,
              "bundle_sha256": digest(bundle_bytes), "python": platform.python_version(),
              "runner_sha256": digest(Path(__file__).read_bytes()),
              "decode_module_sha256": digest(Path(sys.modules[decode_batch.__module__].__file__).read_bytes()),
              "helper_sha256": digest(helper.read_bytes()) if helper is not None else None,
              "validation": {"case_count": len(loaded), "cases": []},
              "benchmark": {"case": "demo", "repeats": repeats, "warmups_per_mode": 1,
                            "timed": "single-job batch file assembly, process/pipeline startup, parsing, decoding, readback, diagnostics and exact I420 byte comparison",
                            "excluded": "compilation, PCM packing, network transfer, reference loading, capture rendering",
                            "order": "four modes rotated by one position each repetition", "modes": {}}}

    def save():
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for name, variants in loaded.items():
        validated = {"name": name, "variants": {}}
        observed = {}
        for variant, entry in variants.items():
            modes = {}
            for selective in (False, True):
                result, actual, counters = execute(entry, backend, selective)
                named = check_named(result, entry)
                if not selective:
                    observed[variant] = named
                elif named != observed[variant]:
                    raise AssertionError("Full and selective named outputs differ")
                modes["outputs" if selective else "all"] = {
                    **counters, "sha256": digest(actual), "named_outputs": len(named), "status": "pass"}
                if name == "demo" and variant == "optimized" and selective:
                    report["capture"] = save_capture(output, result, actual, entry, backend, digest(bundle_bytes))
            validated["variants"][variant] = modes
            print("PASS validate %s/%s: full + selected I420 bytes and %d named outputs" %
                  (name, variant, len(entry["named"])), flush=True)
        if observed["baseline"] != observed["optimized"]:
            raise AssertionError("Baseline and optimized named states differ: " + name)
        validated.update(status="pass", paired_named_outputs_equal=True)
        report["validation"]["cases"].append(validated)
        save()

    modes = [("baseline-all", "baseline", False), ("baseline-outputs", "baseline", True),
             ("optimized-all", "optimized", False), ("optimized-outputs", "optimized", True)]
    samples = {name: [] for name, _, _ in modes}
    for iteration in range(repeats + 1):
        offset = iteration % len(modes)
        for name, variant, selective in modes[offset:] + modes[:offset]:
            started = time.perf_counter()
            _, _, counters = execute(loaded["demo"][variant], backend, selective)
            elapsed = (time.perf_counter() - started) * 1000
            if iteration:
                samples[name].append(elapsed)
                report["benchmark"]["modes"][name] = {
                    **counters, "variant": variant, "readback": "outputs" if selective else "all",
                    "samples_ms": samples[name], "median_ms": statistics.median(samples[name]),
                    "min_ms": min(samples[name]), "max_ms": max(samples[name]), "status": "pass"}
                save()
            print("PASS %s %s: %.2f ms" % (name, "warmup" if not iteration else "run %d" % iteration, elapsed), flush=True)
    timing = report["benchmark"]["modes"]
    report["benchmark"]["speedups"] = {
        "prediction_only_full_readback": timing["baseline-all"]["median_ms"] / timing["optimized-all"]["median_ms"],
        "prediction_only_named_readback": timing["baseline-outputs"]["median_ms"] / timing["optimized-outputs"]["median_ms"],
        "readback_only_baseline": timing["baseline-all"]["median_ms"] / timing["baseline-outputs"]["median_ms"],
        "readback_only_optimized": timing["optimized-all"]["median_ms"] / timing["optimized-outputs"]["median_ms"],
        "combined": timing["baseline-all"]["median_ms"] / timing["optimized-outputs"]["median_ms"]}
    report["status"] = "pass"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Directory for results.json and actual captures")
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--backend", choices=("nvdec", "ffmpeg", "vaapi", "openh264"), default="nvdec")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    benchmark(args.bundle, args.output, args.backend, args.helper, args.repeats)


if __name__ == "__main__":
    main()
