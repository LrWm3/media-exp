#!/usr/bin/env python3
"""Classify official MNIST 3s and 8s using actual H.264 decoder inference."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

from codec import decode
from compiler import CompiledClassifier
from data import DEFAULT_CACHE, load
from network import FrozenNetwork
from verify import extract, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "model/weights.json")
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--samples", type=int, default=16, help="First N test digits; 0 means all 3s and 8s")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("out/mnist.h264"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.samples < 0 or not 1 <= args.batch_size <= 8:
        parser.error("samples must be nonnegative; batch size must be 1..8")
    model = FrozenNetwork.load(args.model)
    digits = load("test", args.data)
    if args.samples:
        digits = digits[:args.samples]
    start = perf_counter()
    compiled = CompiledClassifier(model, args.batch_size)
    compile_seconds = perf_counter()-start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch_path = args.output.with_suffix(".batch.h264")
    results, preview = [], []
    decode_seconds, verify_seconds, seed_seconds = 0.0, 0.0, 0.0
    raw_hash = hashlib.sha256()
    stream_hash = hashlib.sha256()
    total_started = perf_counter()
    try:
        with args.output.open("wb") as destination:
            for start in range(0, len(digits), args.batch_size):
                batch = digits[start:start+args.batch_size]
                images = [digit.binary() for digit in batch]
                tick = perf_counter()
                stream = compiled.encode(images)
                seed_seconds += perf_counter()-tick
                destination.write(stream)
                batch_path.write_bytes(stream)
                stream_hash.update(stream)
                tick = perf_counter()
                raw = decode(batch_path, args.decoder)
                decode_seconds += perf_counter()-tick
                raw_hash.update(raw)
                tick = perf_counter()
                records = extract(raw, images, compiled)
                if args.verify:
                    verify(records, images, model)
                verify_seconds += perf_counter()-tick
                for digit, record in zip(batch, records):
                    prediction = (3, 8)[record["bit"]]
                    results.append({"test_index": digit.index, "label": digit.label, "prediction": prediction,
                                    "correct": digit.label == prediction, "hidden": record["hidden"]})
                    if len(preview) < 16:
                        preview.append((digit, record))
                if start == 0 or len(results) % 128 == 0 or len(results) == len(digits):
                    accuracy = sum(r["correct"] for r in results)/len(results)
                    print(f"{args.decoder}: {len(results)}/{len(digits)} classified; accuracy {accuracy:.2%}", flush=True)
    finally:
        batch_path.unlink(missing_ok=True)
    confusion = [[sum(r["label"] == label and r["prediction"] == prediction for r in results)
                  for prediction in (3, 8)] for label in (3, 8)]
    program = compiled.program
    report = {"decoder": args.decoder, "dataset": "official MNIST test split, classes 3 and 8, original order",
              "samples": len(digits), "classes": [3, 8], "confusion_matrix_rows_true_columns_predicted": confusion,
              "correct": sum(r["correct"] for r in results), "accuracy": sum(r["correct"] for r in results)/len(results),
              "batch_capacity": args.batch_size, "batches": (len(digits)+args.batch_size-1)//args.batch_size,
              "pictures_per_batch": program.pictures, "reference_copies_per_batch": len(program.refreshes),
              "maximum_reference_age": max(frame-s.frame for frame, op in enumerate(program.operations, 1) for s in op.sources),
              "maximum_motion_pixels": max(max(abs(s.dx), abs(s.dy)) for op in program.operations for s in op.sources),
              "stream_bytes": args.output.stat().st_size, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "stream_sha256": stream_hash.hexdigest(), "decoded_raw_sha256": raw_hash.hexdigest(),
              "instruction_compile_seconds": compile_seconds, "seed_encode_seconds": seed_seconds,
              "decode_and_readback_seconds": decode_seconds, "extract_and_verify_seconds": verify_seconds,
              "batch_loop_seconds": perf_counter()-total_started, "verified": args.verify, "predictions": results}
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2)+"\n")
    args.output.with_suffix(".program.json").write_text(json.dumps(asdict(program), indent=2)+"\n")
    print(f"Accuracy {report['accuracy']:.2%} ({report['correct']}/{len(digits)}); confusion {confusion}")
    print(f"{program.pictures} pictures per {args.batch_size}-digit batch; decode + readback {decode_seconds:.3f}s")
    if args.preview:
        from capture import render
        render(preview, args.output, args.decoder)
        print(f"Capture: {args.output.with_suffix('.gif')}")


if __name__ == "__main__":
    main()
