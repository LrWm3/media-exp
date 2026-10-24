#!/usr/bin/env python3
"""Train a binary-weight, binary-activation 3-versus-8 classifier in PyTorch."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import DEFAULT_CACHE, load, training_split


def binary_weight(value):
    hard = torch.where(value >= 0, 1.0, -1.0)
    return value + (hard - value).detach()


def binary_activation(value):
    hard = (value >= 0).to(value.dtype)
    soft = torch.sigmoid(value)
    return soft + (hard - soft).detach()


class Classifier(nn.Module):
    def __init__(self, channels=8, pool_threshold=1):
        super().__init__()
        self.pool_threshold = pool_threshold
        self.weights = nn.Parameter(torch.empty(channels, 196).uniform_(-1, 1))
        self.thresholds = nn.Parameter(torch.zeros(channels))
        self.mix = nn.Parameter(torch.empty(1, channels).uniform_(-1, 1))
        self.mix_threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, image):
        cells = (F.avg_pool2d(image, 2) * 4 >= self.pool_threshold).float().flatten(1)
        features = binary_activation(F.linear(cells, binary_weight(self.weights)) - self.thresholds)
        return F.linear(features, binary_weight(self.mix)).flatten() - self.mix_threshold

    def freeze(self):
        with torch.no_grad():
            return {"format": "binary-mnist-3-vs-8-v1", "input_size": 28,
                    "input_threshold": 128, "pool_size": 2, "pool_threshold": self.pool_threshold,
                    "classes": [3, 8], "activation": "sum >= integer threshold",
                    "weights": binary_weight(self.weights).int().tolist(),
                    "thresholds": torch.ceil(self.thresholds).int().tolist(),
                    "mix_weights": binary_weight(self.mix).int().flatten().tolist(),
                    "mix_threshold": int(torch.ceil(self.mix_threshold))}


def tensors(digits):
    pixels = np.frombuffer(b"".join(d.pixels for d in digits), dtype=np.uint8).reshape(-1, 1, 28, 28)
    return torch.tensor(pixels >= 128, dtype=torch.float32), torch.tensor([d.label == 8 for d in digits], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--data", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path("model/weights.json"))
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.channels <= 16:
        parser.error("positive steps and 1..16 channels required")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train, validation = training_split(load("train", args.data), args.seed)
    tx, ty = tensors(train)
    vx, vy = tensors(validation)
    model = Classifier(args.channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    best_error, best_step, best_state = float("inf"), 0, None
    history = []
    started = perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train), (128,))
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(tx[indices]) / 3, ty[indices])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.weights.clamp_(-1, 1)
            model.mix.clamp_(-1, 1)
            model.thresholds.clamp_(-196, 197)
            model.mix_threshold.clamp_(-args.channels, args.channels + 1)
        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                error = ((model(vx) >= 0) != vy.bool()).float().mean().item()
            history.append({"step": step, "loss": loss.item(), "validation_error": error})
            if error < best_error:
                best_error, best_step, best_state = error, step, copy.deepcopy(model.state_dict())
            print(f"step {step:4d} validation accuracy {1-error:.2%} best {1-best_error:.2%}", flush=True)
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model.freeze(), indent=2) + "\n")
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    # Test split is deliberately not accessed during architecture/checkpoint selection.
    report = {"torch_version": torch.__version__, "seed": args.seed, "steps": args.steps,
              "channels": args.channels, "optimizer": "Adam", "learning_rate": 0.008, "batch_size": 128,
              "pool_size": 2, "pool_threshold": 1, "loss_logit_divisor": 3,
              "train_images": len(train), "validation_images": len(validation),
              "validation_per_class": 500, "best_step": best_step, "validation_accuracy": 1-best_error,
              "selection": "lowest validation error; earliest checkpoint wins ties",
              "train_indices_sha256": hashlib.sha256(json.dumps([d.index for d in train]).encode()).hexdigest(),
              "validation_indices_sha256": hashlib.sha256(json.dumps([d.index for d in validation]).encode()).hexdigest(),
              "elapsed_seconds": perf_counter()-started, "history": history}
    args.output.with_name("training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
