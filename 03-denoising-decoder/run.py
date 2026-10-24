#!/usr/bin/env python3
"""Run frozen denoising inference through a video decoder, optionally verify it."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from capture import render
from codec import decode
from compiler import compile_batch
from data import error_rate, majority, samples
from network import FrozenNetwork
from verify import extract, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "model/weights.json")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("out/denoising.h264"))
    parser.add_argument("--verify", action="store_true", help="Compare every feature/output bit with independent integer inference")
    parser.add_argument("--preview", action="store_true", help="Capture clean/noisy/decoded comparisons and features")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if not 0 <= args.noise <= 1:
        parser.error("noise must be in [0, 1]")
    try:
        model = FrozenNetwork.load(args.model)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    clean, noisy = samples(args.samples, args.seed, probability=args.noise)
    started = perf_counter()
    stream, program = compile_batch(noisy, model)
    compile_seconds = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stream)
    started = perf_counter()
    raw = decode(args.output, args.decoder)
    decode_seconds = perf_counter() - started
    decoded, features = extract(raw, noisy, program)
    if args.verify:
        verify(decoded, features, noisy, model)
        print(f"PASS ({args.decoder}): every feature and output bit matches for {args.samples} images")
    report = {
        "decoder": args.decoder, "samples": args.samples, "seed": args.seed,
        "noise_probability": args.noise, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "stream_sha256": hashlib.sha256(stream).hexdigest(), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "stream_bytes": len(stream), "pictures_per_image": program.pictures,
        "prediction_pictures": len(program.operations), "reference_copies_per_image": len(program.refreshes),
        "maximum_reference_age": max(frame - source.frame for frame, op in enumerate(program.operations, 1) for source in op.sources),
        "noisy_error": error_rate(noisy, clean), "decoded_error": error_rate(decoded, clean),
        "majority_error": error_rate([majority(image) for image in noisy], clean),
        "compile_seconds": compile_seconds, "decode_and_readback_seconds": decode_seconds,
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {"pictures_per_image": program.pictures, "features": program.features,
                "output": program.output, "refreshes": program.refreshes,
                "operations": [{"frame": frame, **asdict(op)} for frame, op in enumerate(program.operations, 1)]}
    args.output.with_suffix(".program.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(stream):,} bytes; {program.pictures} pictures/image")
    print(f"Pixel error: noisy {report['noisy_error']:.4%}; decoder {report['decoded_error']:.4%}; majority {report['majority_error']:.4%}")
    print(f"Compile {compile_seconds:.3f}s; decode + readback {decode_seconds:.3f}s")
    if args.preview:
        render(clean, noisy, decoded, features, args.output, args.decoder)
        print(f"Capture: {args.output.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
