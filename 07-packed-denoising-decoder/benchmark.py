#!/usr/bin/env python3
"""Validate and time portable packed-denoiser bundles on Python 3.8 or later.

Only the standard library and decoder_core.decode are required. No compiler,
model, dataset generator, or inference library is imported on the decode host.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

for parent in Path(__file__).resolve().parents:
    if (parent / "decoder_core" / "decode.py").is_file():
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
        raise ValueError("Bundle checksum mismatch: " + relative)
    return data


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def geometry(manifest, coded_width=None, coded_height=None):
    """Normalize compiler metadata without importing the compiler."""
    width, height = manifest["width"], manifest["height"]
    guard = manifest.get("guard", 16)
    stride = coded_width or manifest.get("coded_width", width + 2 * guard)
    rows = coded_height or manifest.get("coded_height", height + 2 * guard)
    pictures = manifest["pictures"]
    if any(type(value) is not int or value < 1 for value in
           (width, height, guard, stride, rows, pictures)):
        raise ValueError("Picture dimensions and counts must be positive integers")
    if stride % 2 or rows % 2 or width + 2 * guard > stride or height + 2 * guard > rows:
        raise ValueError("Invalid I420 geometry")
    tiles = manifest.get("tiles", [{"tile": 0, "x": 0, "y": 0,
                                     "width": width, "height": height}])
    tile_map = {}
    for tile in tiles:
        if type(tile["tile"]) is not int or tile["tile"] < 0 or tile["tile"] in tile_map:
            raise ValueError("Invalid or duplicate tile index")
        if any(type(tile[key]) is not int for key in ("x", "y", "width", "height")):
            raise ValueError("Tile coordinates must be integers")
        if min(tile["x"], tile["y"]) < 0 or min(tile["width"], tile["height"]) < 1:
            raise ValueError("Invalid tile rectangle")
        if tile["x"] + tile["width"] > stride - 2 * guard or tile["y"] + tile["height"] > rows - 2 * guard:
            raise ValueError("Tile rectangle exceeds coded canvas")
        tile_map[tile["tile"]] = dict(tile)
    frame_tiles = manifest.get("frame_tiles")
    if frame_tiles is None:
        if len(tiles) != 1:
            raise ValueError("Packed pictures require frame_tiles metadata")
        frame_tiles = {str(frame): [tiles[0]["tile"]] for frame in range(pictures)}
    if set(frame_tiles) != {str(frame) for frame in range(pictures)}:
        raise ValueError("frame_tiles must describe every physical picture")
    for active in frame_tiles.values():
        if not isinstance(active, list) or any(type(tile) is not int for tile in active):
            raise ValueError("Invalid active tile indices")
        if len(active) != len(set(active)) or any(tile not in tile_map for tile in active):
            raise ValueError("Unknown or duplicate active tile")
    regions = manifest.get("output_regions") or {
        name: {"frame": frame, "tile": 0, "x": 0, "y": 0, "width": width, "height": height}
        for name, frame in manifest["output_frames"].items()}
    if set(regions) != set(manifest["output_frames"]):
        raise ValueError("Output region names differ from output_frames")
    for name, region in regions.items():
        frame = region["frame"]
        if type(frame) is not int or not 0 <= frame < pictures or frame != manifest["output_frames"][name]:
            raise ValueError("Invalid output picture index")
        tile = tile_map.get(region["tile"])
        if tile is None or region["tile"] not in frame_tiles[str(frame)]:
            raise ValueError("Output refers to an inactive tile")
        if any(region[key] != tile[key] for key in ("x", "y", "width", "height")):
            raise ValueError("Output region differs from its logical tile")
        if (region["width"], region["height"]) != (width, height):
            raise ValueError("Output dimensions differ from logical image")
    return {"width": width, "height": height, "guard": guard,
            "coded_width": stride, "coded_height": rows,
            "frame_bytes": stride * rows * 3 // 2, "pictures_per_job": pictures,
            "tiles": tiles, "frame_tiles": frame_tiles, "output_regions": regions,
            "output_picture_indices": sorted({region["frame"] for region in regions.values()})}


def padding_spans(shape):
    """Precompute zero-luma spans, including unused tiles and inner gutters."""
    stride, rows, guard = (shape[key] for key in ("coded_width", "coded_height", "guard"))
    tiles = {tile["tile"]: tile for tile in shape["tiles"]}
    result = {}
    for frame, active in shape["frame_tiles"].items():
        spans = []
        for y in range(rows):
            intervals = sorted((tile["x"] + guard, tile["x"] + guard + tile["width"])
                               for tile in (tiles[index] for index in active)
                               if tile["y"] + guard <= y < tile["y"] + guard + tile["height"])
            previous = 0
            for left, right in intervals:
                if left < previous:
                    raise ValueError("Active tile rectangles overlap")
                if left > previous:
                    spans.append((y * stride + previous, y * stride + left))
                previous = right
            if previous < stride:
                spans.append((y * stride + previous, (y + 1) * stride))
        result[int(frame)] = spans
    return result


def validate_frames(frames, shape, selected_frames=None, spans=None):
    """Check every returned I420 byte outside active logical tile interiors."""
    selection = tuple(range(shape["pictures_per_job"])) if selected_frames is None else tuple(selected_frames)
    if len(frames) != len(selection):
        raise AssertionError("Decoded picture count differs from selection")
    spans = padding_spans(shape) if spans is None else spans
    luma_size = shape["coded_width"] * shape["coded_height"]
    neutral = bytes([128]) * (luma_size // 2)
    checked_padding = 0
    for frame, raw in zip(selection, frames):
        if len(raw) != shape["frame_bytes"]:
            raise AssertionError("Decoded I420 frame size mismatch")
        if raw[luma_size:] != neutral:
            raise AssertionError("Nonneutral chroma in picture %d" % frame)
        for start, end in spans[frame]:
            if raw[start:end] != bytes(end - start):
                raise AssertionError("Nonzero padding or unused tile in picture %d" % frame)
            checked_padding += end - start
    return {"padding_bytes": checked_padding, "chroma_bytes": len(selection) * luma_size // 2}


def extract_region(raw_frame, region, shape):
    stride, guard = shape["coded_width"], shape["guard"]
    left, top = region["x"] + guard, region["y"] + guard
    return b"".join(raw_frame[y * stride + left:y * stride + left + region["width"]]
                    for y in range(top, top + region["height"]))


def counters(result):
    returned = sum(len(frame) for frames in result.frames for frame in frames)
    return {"decoder_sessions": 1, "decoded_pictures": result.decoded_frames,
            "copied_pictures": result.copied_frames,
            "decoded_i420_bytes": result.decoded_frames * result.frame_bytes,
            "copied_i420_bytes": result.copied_frames * result.frame_bytes,
            "returned_pictures": sum(len(frames) for frames in result.frames),
            "returned_i420_bytes": returned}


def benchmark(bundle_path, output, backend="nvdec", repeats=5, helper=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    if helper is not None and backend != "nvdec":
        raise ValueError("--helper applies only to NVDEC")
    if backend == "nvdec":
        helper = Path(helper).resolve() if helper else _nvdec_helper()
        os.environ["H264_NVDEC_DECODER"] = str(helper)
    if not available(backend):
        raise RuntimeError("Requested backend is unavailable: " + backend)
    root, output = Path(bundle_path).resolve(), Path(output).resolve()
    bundle_bytes = (root / "bundle.json").read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle.get("format") != "h264-packed-denoiser-bundle-0.1":
        raise ValueError("Unsupported packed denoiser bundle")
    if bundle["job_count"] != 8 or len(bundle["inputs"]) != 8:
        raise ValueError("Comparison requires the same eight input images")
    if "baseline" not in bundle["variants"] or not bundle["variants"]:
        raise ValueError("Bundle must include baseline")
    # These immutable references are loaded before any measurements.
    oracles = []
    for index, item in enumerate(bundle["inputs"]):
        if item["index"] != index:
            raise ValueError("Input indices must be consecutive")
        checked_file(root, item["path"], item["sha256"])
        oracles.append({name: checked_file(root, info["path"], info["sha256"])
                        for name, info in item["named_outputs"].items()})
    workloads = {}
    for name, variant in bundle["variants"].items():
        manifest = json.loads(checked_file(root, variant["manifest"], variant["manifest_sha256"]))
        shape = geometry(manifest, variant["coded_width"], variant["coded_height"])
        tail = checked_file(root, variant["tail"], variant["tail_sha256"])
        if len(tail) != variant["tail_bytes_per_job"]:
            raise ValueError("Prediction tail size mismatch")
        if len(variant["jobs"]) != bundle["job_count"]:
            raise ValueError("Variant input count mismatch")
        paths, references = [], []
        for index, job in enumerate(variant["jobs"]):
            if job["index"] != index or job["input_sha256"] != bundle["inputs"][index]["sha256"]:
                raise ValueError("Variants must use identical input ordering")
            stream = checked_file(root, job["stream"], job["stream_sha256"])
            prefix = checked_file(root, job["input_prefix"], job["input_prefix_sha256"])
            if stream != prefix + tail or len(prefix) != job["input_bytes"] or len(stream) != job["stream_bytes"]:
                raise ValueError("Stream does not contain the immutable input and prediction tail")
            raw = checked_file(root, job["expected"], job["expected_sha256"])
            if len(raw) != shape["pictures_per_job"] * shape["frame_bytes"]:
                raise ValueError("Reference picture count mismatch")
            if set(oracles[index]) != set(shape["output_regions"]):
                raise ValueError("Named outputs differ from oracle")
            if any(len(values) != shape["width"] * shape["height"] for values in oracles[index].values()):
                raise ValueError("Oracle dimensions mismatch")
            paths.append(root / job["stream"])
            references.append(tuple(raw[start:start + shape["frame_bytes"]]
                                    for start in range(0, len(raw), shape["frame_bytes"])))
        workloads[name] = {"shape": shape, "paths": paths, "references": references}
    report = {"format": "h264-packed-denoiser-results-0.1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "backend": backend,
              "bundle_sha256": digest(bundle_bytes), "job_count": bundle["job_count"],
              "model_sha256": bundle["model_sha256"], "input_seed": bundle["input_seed"],
              "input_sha256": [item["sha256"] for item in bundle["inputs"]],
              "helper_sha256": digest(helper.read_bytes()) if helper else None,
              "decode_module_sha256": digest(Path(sys.modules[decode_batch.__module__].__file__).read_bytes()),
              "benchmark": {"repeats": repeats, "warmups_per_variant_and_mode": 1,
                            "primary_mode": "batch-outputs", "batch_jobs": bundle["job_count"],
                            "timed": "batch assembly and file I/O, one process/pipeline startup, parsing, decode, actual surface copies/readback, diagnostics, exact returned I420 comparison and named-output extraction/comparison",
                            "excluded": "network, compilation, input packing, reference loading, full padding/chroma validation and capture writing",
                            "order": "rotate variant/mode order each repetition after one warmup each",
                            "output_selection": "all named feature and final output pictures; aliases deduplicated",
                            "copy_accounting": "NVDEC copies selected pictures; other backends copy all pictures before Python selection"},
              "variants": {}}

    def save():
        write_json(output / "results.json", report)

    save()
    for name, workload in workloads.items():
        shape, variant = workload["shape"], bundle["variants"][name]
        result = decode_batch(workload["paths"], backend, width=shape["coded_width"],
                              height=shape["coded_height"], pictures_per_job=shape["pictures_per_job"])
        if len(result.frames) != bundle["job_count"]:
            raise AssertionError("Full validation job count mismatch")
        entry = {"geometry": shape, "costs": variant["costs"],
                 "tail_bytes_per_job": variant["tail_bytes_per_job"], "tail_sha256": variant["tail_sha256"],
                 "input_bytes_per_job": [job["input_bytes"] for job in variant["jobs"]],
                 "input_prefix_sha256": [job["input_prefix_sha256"] for job in variant["jobs"]],
                 "stream_bytes_per_batch": sum(job["stream_bytes"] for job in variant["jobs"]),
                 "validation": {"status": "pass", **counters(result), "padding_bytes": 0,
                                "chroma_bytes": 0, "named_outputs": 0, "jobs": []}, "modes": {}}
        spans = padding_spans(shape)
        for index, (frames, reference) in enumerate(zip(result.frames, workload["references"])):
            if tuple(frames) != reference:
                raise AssertionError("Full decoded I420 differs: %s job %d" % (name, index))
            checks = validate_frames(frames, shape, spans=spans)
            for key, count in checks.items():
                entry["validation"][key] += count
            named = {}
            for label, region in shape["output_regions"].items():
                values = extract_region(frames[region["frame"]], region, shape)
                if values != oracles[index][label]:
                    raise AssertionError("Frozen-network oracle differs: %s/%d/%s" % (name, index, label))
                capture = Path("captures") / name / ("%03d" % index) / (label + ".pgm")
                (output / capture).parent.mkdir(parents=True, exist_ok=True)
                # Display binary images at full contrast; retain native 0/1 checksum.
                (output / capture).write_bytes(("P5\n%d %d\n255\n" % (shape["width"], shape["height"])).encode()
                                                + bytes(value * 255 for value in values))
                named[label] = {"status": "pass", "sha256": digest(values), "capture": str(capture), **region}
                entry["validation"]["named_outputs"] += 1
            entry["validation"]["jobs"].append({"index": index, "status": "pass",
                                                   "i420_sha256": digest(b"".join(frames)), "named_outputs": named})
        report["variants"][name] = entry
        save()
        print("PASS %s full validation: %d pictures, every named output, zero padding, neutral chroma" %
              (name, result.decoded_frames), flush=True)

    def execute(name, mode):
        workload = workloads[name]
        shape = workload["shape"]
        selection = tuple(shape["output_picture_indices"]) if mode == "batch-outputs" else None
        result = decode_batch(workload["paths"], backend, selected_frames=selection,
                              width=shape["coded_width"], height=shape["coded_height"],
                              pictures_per_job=shape["pictures_per_job"])
        if len(result.frames) != bundle["job_count"]:
            raise AssertionError("Timed batch job count mismatch")
        positions = {frame: index for index, frame in enumerate(result.selected_frames)}
        for index, (frames, reference) in enumerate(zip(result.frames, workload["references"])):
            if tuple(frames) != tuple(reference[frame] for frame in result.selected_frames):
                raise AssertionError("Timed decoded I420 differs: " + name)
            for label, region in shape["output_regions"].items():
                if extract_region(frames[positions[region["frame"]]], region, shape) != oracles[index][label]:
                    raise AssertionError("Timed oracle differs: " + name + "/" + label)
        return counters(result)

    runs = [(name, mode) for name in workloads for mode in ("batch-outputs", "batch-all")]
    samples = {run: [] for run in runs}
    for iteration in range(repeats + 1):
        offset = iteration % len(runs)
        for name, mode in runs[offset:] + runs[:offset]:
            started = time.perf_counter()
            actual = execute(name, mode)
            elapsed = (time.perf_counter() - started) * 1000
            if iteration:
                values = samples[name, mode]
                values.append(elapsed)
                report["variants"][name]["modes"][mode] = {
                    "status": "pass", **actual, "samples_ms": values,
                    "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}
                save()
            print("PASS %s %s %s: %.2f ms" %
                  (name, mode, "warmup" if not iteration else "run %d" % iteration, elapsed), flush=True)
    for name, entry in report["variants"].items():
        for mode, value in entry["modes"].items():
            value["speedup_vs_baseline"] = report["variants"]["baseline"]["modes"][mode]["median_ms"] / value["median_ms"]
    report["status"] = "pass"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("ffmpeg", "openh264", "vaapi", "nvdec"), default="nvdec")
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = benchmark(args.bundle, args.output, args.backend, args.repeats, args.helper)
    print(json.dumps({"status": result["status"], "backend": result["backend"],
                      "medians_ms": {name: {mode: value["median_ms"] for mode, value in entry["modes"].items()}
                                     for name, entry in result["variants"].items()}}), flush=True)


if __name__ == "__main__":
    main()
