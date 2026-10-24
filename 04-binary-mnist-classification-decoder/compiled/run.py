#!/usr/bin/env python3
"""Classify original MNIST digits through the shared H.264 compiler."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from classifier import MODEL_PATH, CompiledClassifier, FrozenNetwork, fixtures
from classifier_original import load
from decoder_core.decode import decode


def main():
    data = load("data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--data", type=Path, default=data.DEFAULT_CACHE)
    parser.add_argument("--fixtures", action="store_true", help="Use checked-in offline demo inputs")
    parser.add_argument("--samples", type=int, default=16, help="0 means every selected test digit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out/mnist.h264")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-checkpoint", action="store_true", help="Also compare actual PyTorch checkpoint activations")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.samples < 0 or not 1 <= args.batch_size <= 8:
        parser.error("samples must be nonnegative; batch size must be 1..8")
    model = FrozenNetwork.load(args.model)
    digits = fixtures() if args.fixtures else data.load("test", args.data)
    if args.samples:
        digits = digits[:args.samples]
    if args.verify_checkpoint:
        import torch
        train = load("train")
        torch.set_num_threads(2)
        checkpoint = train.Classifier(channels=len(model.weights), pool_threshold=model.pool_threshold)
        checkpoint.load_state_dict(torch.load(args.model.with_suffix(".pt"), weights_only=True))
        checkpoint.eval()
    started = perf_counter()
    compiled = CompiledClassifier(model, args.batch_size)
    compile_seconds = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".program.json").write_text(json.dumps(compiled.program.manifest(), indent=2) + "\n")
    batch_path = args.output.with_suffix(".batch.h264")
    results, preview = [], []
    seed_seconds = decode_seconds = verify_seconds = 0.0
    stream_hash, raw_hash = hashlib.sha256(), hashlib.sha256()
    verify = load("verify").verify
    try:
        with args.output.open("wb") as destination:
            for start in range(0, len(digits), args.batch_size):
                batch = digits[start:start + args.batch_size]
                images = [digit.binary() for digit in batch]
                tick = perf_counter()
                stream = compiled.encode(images)
                seed_seconds += perf_counter() - tick
                destination.write(stream)
                stream_hash.update(stream)
                batch_path.write_bytes(stream)
                tick = perf_counter()
                raw = decode(batch_path, args.decoder)
                records = compiled.read(raw, len(batch))
                decode_seconds += perf_counter() - tick
                raw_hash.update(raw)
                tick = perf_counter()
                if args.verify:
                    verify(records, images, model)
                if args.verify_checkpoint:
                    with torch.no_grad():
                        tensor, _ = train.tensors(batch)
                        cells = (torch.nn.functional.avg_pool2d(tensor, 2) * 4 >= checkpoint.pool_threshold).float()
                        hidden = train.binary_activation(torch.nn.functional.linear(
                            cells.flatten(1), train.binary_weight(checkpoint.weights)) - checkpoint.thresholds).int().tolist()
                        bits = (checkpoint(tensor) >= 0).int().tolist()
                        pools = cells.int().squeeze(1).tolist()
                    expected = [{"bit": bit, "hidden": neuron, "pooled": pool}
                                for bit, neuron, pool in zip(bits, hidden, pools)]
                    if records != expected:
                        raise AssertionError(f"Decoder differs from PyTorch checkpoint in batch starting at {start}")
                verify_seconds += perf_counter() - tick
                for digit, record in zip(batch, records):
                    prediction = (3, 8)[record["bit"]]
                    results.append({"test_index": digit.index, "label": digit.label,
                                    "prediction": prediction, "correct": prediction == digit.label,
                                    "hidden": record["hidden"]})
                    if len(preview) < 16:
                        preview.append((digit, record))
                if start == 0 or len(results) % 128 == 0 or len(results) == len(digits):
                    print(f"{args.decoder}: {len(results)}/{len(digits)} classified", flush=True)
    finally:
        batch_path.unlink(missing_ok=True)
    correct = sum(record["correct"] for record in results)
    confusion = [[sum(record["label"] == label and record["prediction"] == prediction for record in results)
                  for prediction in (3, 8)] for label in (3, 8)]
    report = {
        "implementation": "shared-h264-compiler", "decoder": args.decoder,
        "dataset": "checked-in MNIST test fixture" if args.fixtures else "official MNIST test split, classes 3 and 8, original order",
        "samples": len(digits), "classes": [3, 8], "verified": args.verify,
        "checkpoint_verified": args.verify_checkpoint,
        "correct": correct, "accuracy": correct / len(digits),
        "confusion_matrix_rows_true_columns_predicted": confusion,
        "batch_capacity": args.batch_size, "batches": (len(digits) + args.batch_size - 1) // args.batch_size,
        "pictures_per_batch": compiled.program.pictures, "stream_bytes": args.output.stat().st_size,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "instruction_tail_sha256": hashlib.sha256(compiled.program.tail).hexdigest(),
        "stream_sha256": stream_hash.hexdigest(), "decoded_raw_sha256": raw_hash.hexdigest(),
        "instruction_compile_seconds": compile_seconds, "seed_encode_seconds": seed_seconds,
        "decode_and_readback_seconds": decode_seconds, "verify_seconds": verify_seconds,
        "predictions": results,
    }
    if args.verify_checkpoint:
        report["checkpoint_sha256"] = hashlib.sha256(args.model.with_suffix(".pt").read_bytes()).hexdigest()
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Accuracy {report['accuracy']:.2%} ({correct}/{len(digits)}); confusion {confusion}")
    print(f"{compiled.program.pictures} pictures per batch; wrote {args.output}")
    if args.preview:
        load("capture").render(preview, args.output, args.decoder)


if __name__ == "__main__":
    main()
