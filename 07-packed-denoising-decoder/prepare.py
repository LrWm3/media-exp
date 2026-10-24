#!/usr/bin/env python3
"""Prepare identical frozen-denoiser inputs for baseline and packed decoders.

The resulting bundle and decoder_core suffice for portable benchmark.py runs;
only this preparation step imports the compiler and frozen-network oracle.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from kernel import MODEL_PATH, FrozenNetwork, compile_baseline, h264, load
from reference import denoise
from benchmark import digest, geometry, extract_region, padding_spans, validate_frames, write_json
from decoder_core.decode import available, decode_batch


ROOT = Path(__file__).resolve().parent


def plane(board):
    return bytes(value for row in board for value in row)


def save_bytes(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(relative), "sha256": digest(data), "bytes": len(data)}


def prepare(output, tiles=(2, 4, 8), decoders=("ffmpeg", "openh264", "vaapi")):
    output = Path(output).resolve()
    if len(set(tiles)) != len(tiles) or any(type(count) is not int or count < 1 for count in tiles):
        raise ValueError("Tile counts must be unique positive integers")
    if not available("ffmpeg"):
        raise RuntimeError("FFmpeg is required for full I420 reference preparation")
    backends = tuple(dict.fromkeys(("ffmpeg",) + tuple(decoders)))
    ready = tuple(backend for backend in backends if available(backend))
    model = FrozenNetwork.load(MODEL_PATH)
    clean, images = load("data").samples(8, seed=200000)
    expected = [denoise(image, model) for image in images]
    sources = sorted([*ROOT.glob("*.py"), *(ROOT.parent / "05-h264-compiler" / "h264").glob("*.py"),
                      ROOT.parent / "decoder_core" / "bitstream.py", ROOT.parent / "decoder_core" / "decode.py",
                      ROOT.parent / "03-denoising-decoder" / "network.py",
                      ROOT.parent / "03-denoising-decoder" / "data.py",
                      ROOT.parent / "03-denoising-decoder" / "compiled" / "denoiser.py"])
    report = {"format": "h264-packed-denoiser-bundle-0.1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "job_count": 8,
              "input_seed": 200000, "input_description": "the original deterministic noisy binary shapes",
              "model_sha256": digest(MODEL_PATH.read_bytes()),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "source_sha256": {str(path.relative_to(ROOT.parent)): digest(path.read_bytes()) for path in sources},
              "decoders": {backend: "available" if backend in ready else "unavailable" for backend in backends},
              "inputs": [], "variants": {}}
    for index, (image, target, named) in enumerate(zip(images, clean, expected)):
        item = {"index": index, **save_bytes(output, Path("inputs") / ("%03d.u8" % index), plane(image)),
                "clean": save_bytes(output, Path("inputs") / ("%03d-clean.u8" % index), plane(target)),
                "named_outputs": {}}
        for name, board in named.items():
            item["named_outputs"][name] = save_bytes(output, Path("oracle") / ("%03d" % index) / (name + ".u8"), plane(board))
        report["inputs"].append(item)
    started = time.perf_counter()
    baseline = compile_baseline(model)
    lowering_ms = (time.perf_counter() - started) * 1000
    report["baseline_lowering_ms"] = lowering_ms
    variants = [("baseline", None)] + [("packed-%d" % count, count) for count in tiles]
    for name, count in variants:
        started = time.perf_counter()
        program = baseline if count is None else h264.pack(baseline, tiles=count)
        packing_ms = 0 if count is None else (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        tail = program.tail
        tail_ms = (time.perf_counter() - started) * 1000
        manifest = program.manifest()
        shape = geometry(manifest, program.writer.width, program.writer.height)
        if shape["output_picture_indices"] != list(program.output_picture_indices):
            raise AssertionError("Output selection disagrees with program")
        manifest_path = Path(name) / "program.json"
        write_json(output / manifest_path, manifest)
        saved_tail = save_bytes(output, Path(name) / "prediction.h264", tail)
        variant = {"tiles_requested": count, "manifest": str(manifest_path),
                   "manifest_sha256": digest((output / manifest_path).read_bytes()),
                   "width": program.width, "height": program.height,
                   "coded_width": program.writer.width, "coded_height": program.writer.height,
                   "pictures_per_job": program.pictures, "frame_bytes": shape["frame_bytes"],
                   "output_picture_indices": list(program.output_picture_indices),
                   "output_regions": shape["output_regions"], "costs": program.costs,
                   "lowering_ms": lowering_ms, "packing_ms": packing_ms, "first_tail_ms": tail_ms,
                   "tail": saved_tail["path"], "tail_sha256": saved_tail["sha256"],
                   "tail_bytes_per_job": len(tail), "jobs": [], "decoders": {}}
        paths = []
        for index, (image, named) in enumerate(zip(images, expected)):
            if baseline.reference(image) != named:
                raise AssertionError("Compiled graph differs from frozen-network oracle")
            started = time.perf_counter()
            prefix = program.encode_inputs(image)
            input_ms = (time.perf_counter() - started) * 1000
            # Prove program encode uses precisely the frozen tail and PCM prefix.
            stream = program.encode(image)
            if stream != prefix + tail:
                raise AssertionError("Prediction tail changed while encoding an input")
            saved_input = save_bytes(output, Path(name) / "inputs" / ("%03d.h264" % index), prefix)
            saved_stream = save_bytes(output, Path(name) / "streams" / ("%03d.h264" % index), stream)
            paths.append(output / saved_stream["path"])
            variant["jobs"].append({"index": index, "input_sha256": report["inputs"][index]["sha256"],
                                    "input_prefix": saved_input["path"], "input_prefix_sha256": saved_input["sha256"],
                                    "input_bytes": len(prefix), "input_packing_ms": input_ms,
                                    "stream": saved_stream["path"], "stream_sha256": saved_stream["sha256"],
                                    "stream_bytes": len(stream)})
        options = {"width": program.writer.width, "height": program.writer.height,
                   "pictures_per_job": program.pictures}
        reference = decode_batch(paths, "ffmpeg", **options)
        if len(reference.frames) != len(images):
            raise AssertionError("Reference batch job count differs")
        spans = padding_spans(shape)
        padding_bytes, chroma_bytes = 0, 0
        for index, (frames, named) in enumerate(zip(reference.frames, expected)):
            raw = b"".join(frames)
            if program.read(raw) != named:
                raise AssertionError("Decoded output differs from frozen-network oracle")
            checks = validate_frames(frames, shape, spans=spans)
            padding_bytes += checks["padding_bytes"]
            chroma_bytes += checks["chroma_bytes"]
            for label, region in shape["output_regions"].items():
                if extract_region(frames[region["frame"]], region, shape) != plane(named[label]):
                    raise AssertionError("Portable region extraction differs from frozen-network oracle")
            saved_raw = save_bytes(output, Path(name) / "expected" / ("%03d.i420" % index), raw)
            variant["jobs"][index].update({"expected": saved_raw["path"], "expected_sha256": saved_raw["sha256"],
                                            "expected_bytes": len(raw)})
        for backend in ready:
            decoded = reference if backend == "ffmpeg" else decode_batch(paths, backend, **options)
            if decoded.frames != reference.frames:
                raise AssertionError("%s differs from complete FFmpeg I420 reference for %s" % (backend, name))
            variant["decoders"][backend] = {"status": "pass", "jobs": len(images),
                                             "decoded_pictures": decoded.decoded_frames,
                                             "copied_pictures": decoded.copied_frames,
                                             "i420_bytes": decoded.decoded_frames * decoded.frame_bytes,
                                             "named_outputs": len(images) * len(expected[0]),
                                             "padding_bytes": padding_bytes, "chroma_bytes": chroma_bytes}
        variant["total_pictures"] = len(images) * program.pictures
        variant["total_i420_bytes"] = len(images) * program.pictures * shape["frame_bytes"]
        report["variants"][name] = variant
        print("PASS %s: %d pictures/job, %dx%d coded, %d-byte tail, %s" %
              (name, program.pictures, program.writer.width, program.writer.height, len(tail), ", ".join(ready)), flush=True)
    write_json(output / "bundle.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--tiles", type=int, nargs="*", default=[2, 4, 8],
                        help="packed tile counts; omit values to prepare baseline only")
    parser.add_argument("--decoders", nargs="+", choices=("ffmpeg", "openh264", "vaapi"),
                        default=["ffmpeg", "openh264", "vaapi"])
    args = parser.parse_args()
    report = prepare(args.output, args.tiles, args.decoders)
    print(json.dumps({"job_count": report["job_count"], "variants": list(report["variants"])}), flush=True)


if __name__ == "__main__":
    main()
