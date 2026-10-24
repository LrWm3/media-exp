#!/usr/bin/env python3
"""Train a genuine quantized PyTorch denoiser, then export frozen integer rules."""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import samples


def binary_weight(value):
    hard = torch.where(value >= 0, 1.0, -1.0)
    return value + (hard - value).detach()


def binary_activation(value):
    hard = (value >= 0).to(value.dtype)
    soft = torch.sigmoid(value)
    return soft + (hard - soft).detach()


class Denoiser(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.kernels = nn.Parameter(torch.empty(channels, 1, 3, 3).uniform_(-1, 1))
        self.thresholds = nn.Parameter(torch.zeros(channels))
        self.mix = nn.Parameter(torch.empty(1, channels, 1, 1).uniform_(-1, 1))
        self.mix_threshold = nn.Parameter(torch.tensor(0.0))

    def forward(self, image):
        scores = F.conv2d(image, binary_weight(self.kernels), padding=1)
        features = binary_activation(scores - self.thresholds.view(1, -1, 1, 1))
        return F.conv2d(features, binary_weight(self.mix)) - self.mix_threshold

    def freeze(self):
        with torch.no_grad():
            channels = len(self.kernels)
            return {
                "format": "binary-denoiser-v1",
                "architecture": f"binary 1->{channels} 3x3 convolution, binary {channels}->1 1x1 convolution",
                "padding": "zero",
                "activation": "sum >= integer threshold",
                "kernels": binary_weight(self.kernels).int().reshape(-1, 9).tolist(),
                "thresholds": torch.ceil(self.thresholds).int().tolist(),
                "mix_weights": binary_weight(self.mix).int().flatten().tolist(),
                "mix_threshold": int(torch.ceil(self.mix_threshold)),
            }


def tensor(images):
    return torch.tensor(np.asarray(images), dtype=torch.float32).unsqueeze(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("model/weights.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    train_clean, _ = samples(1024, seed=0)
    valid_clean, valid_noisy = samples(128, seed=100000)
    clean = tensor(train_clean)
    vx, vy = tensor(valid_noisy), tensor(valid_clean)
    model = Denoiser()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    best_error, best_state, best_step = float("inf"), None, 0
    history = []
    start = perf_counter()
    for step in range(1, args.steps + 1):
        target = clean[torch.randint(len(clean), (32,))]
        noise = torch.rand_like(target) < 0.15
        image = torch.where(noise, 1 - target, target)
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(image), target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.kernels.clamp_(-1, 1)
            model.mix.clamp_(-1, 1)
            model.thresholds.clamp_(-9, 10)
            model.mix_threshold.clamp_(-8, 9)
        if step == 1 or step % 50 == 0 or step == args.steps:
            with torch.no_grad():
                error = ((model(vx) >= 0) != vy.bool()).float().mean().item()
            history.append({"step": step, "loss": loss.item(), "validation_error": error})
            if error < best_error:
                best_error, best_step, best_state = error, step, copy.deepcopy(model.state_dict())
            print(f"step {step:4d} loss {loss.item():.4f} validation error {error:.4%} best {best_error:.4%}", flush=True)
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen = model.freeze()
    args.output.write_text(json.dumps(frozen, indent=2) + "\n")
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    # This split is not used for optimization, checkpoint selection, or threshold fitting.
    tx_clean, tx_noisy = samples(256, seed=200000)
    tx, ty = tensor(tx_noisy), tensor(tx_clean)
    with torch.no_grad():
        output = model(tx) >= 0
        test_error = (output != ty.bool()).float().mean().item()
        noisy_error = (tx != ty).float().mean().item()
        majority_error = ((F.conv2d(tx, torch.ones(1, 1, 3, 3), padding=1) >= 5) != ty.bool()).float().mean().item()
    report = {"torch_version": torch.__version__, "seed": args.seed, "steps": args.steps,
              "training_images": 1024, "validation_images": 128, "test_images": 256,
              "train_shape_seeds": [0, 1023], "validation_shape_seeds": [100000, 100127],
              "test_shape_seeds": [200000, 200255], "noise_probability": 0.15,
              "optimizer": "Adam", "learning_rate": 0.015, "batch_size": 32,
              "selection": "lowest fixed validation pixel error", "best_step": best_step,
              "validation_error": best_error, "test_noisy_error": noisy_error,
              "test_model_error": test_error, "test_majority_error": majority_error,
              "elapsed_seconds": perf_counter() - start, "history": history}
    args.output.with_name("training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
