#!/usr/bin/env python3
"""Run the existing denoising weights using the shared H.264 compiler."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from denoiser import MODEL_PATH, FrozenNetwork, compile_model, extract
from denoiser_original import load
from decoder_core.decode import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out/denoising.h264")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-checkpoint", action="store_true", help="Also compare actual PyTorch checkpoint features")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or not 0 <= args.noise <= 1:
        parser.error("positive samples and noise in [0, 1] required")
    model = FrozenNetwork.load(args.model)
    data = load("data")
    clean, noisy = data.samples(args.samples, args.seed, probability=args.noise)
    started = perf_counter()
    program = compile_model(model)
    stream = b"".join(program.encode(image) for image in noisy)
    compile_seconds = perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stream)
    args.output.with_suffix(".program.json").write_text(json.dumps(program.manifest(), indent=2) + "\n")
    started = perf_counter()
    raw = decode(args.output, args.decoder)
    decoded, features = extract(raw, noisy, program, len(model.kernels))
    decode_seconds = perf_counter() - started
    if args.verify:
        load("verify").verify(decoded, features, noisy, model)
    if args.verify_checkpoint:
        import torch
        train = load("train")
        torch.set_num_threads(2)
        checkpoint = train.Denoiser(channels=len(model.kernels))
        checkpoint.load_state_dict(torch.load(args.model.with_suffix(".pt"), weights_only=True))
        checkpoint.eval()
        with torch.no_grad():
            tensor = train.tensor(noisy)
            scores = torch.nn.functional.conv2d(tensor, train.binary_weight(checkpoint.kernels), padding=1)
            expected_features = train.binary_activation(scores - checkpoint.thresholds.view(1, -1, 1, 1)).int().tolist()
            expected_output = (checkpoint(tensor) >= 0).int().squeeze(1).tolist()
        if features != expected_features or decoded != expected_output:
            raise AssertionError("Decoder differs from original PyTorch checkpoint")
    report = {
        "implementation": "shared-h264-compiler", "decoder": args.decoder,
        "samples": args.samples, "seed": args.seed, "noise_probability": args.noise,
        "verified": args.verify, "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "checkpoint_verified": args.verify_checkpoint,
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(), "stream_bytes": len(stream),
        "instruction_tail_sha256": hashlib.sha256(program.tail).hexdigest(),
        "pictures_per_image": program.pictures,
        "noisy_error": data.error_rate(noisy, clean),
        "decoded_error": data.error_rate(decoded, clean),
        "majority_error": data.error_rate([data.majority(image) for image in noisy], clean),
        "compile_and_encode_seconds": compile_seconds,
        "decode_and_readback_seconds": decode_seconds,
    }
    if args.verify_checkpoint:
        report["checkpoint_sha256"] = hashlib.sha256(args.model.with_suffix(".pt").read_bytes()).hexdigest()
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.decoder}: {args.samples} images; {program.pictures} pictures/image; verified={args.verify}")
    print(f"Pixel error: noisy {report['noisy_error']:.4%}; decoded {report['decoded_error']:.4%}")
    print(f"Wrote {args.output}")
    if args.preview:
        load("capture").render(clean, noisy, decoded, features, args.output, args.decoder)


if __name__ == "__main__":
    main()
