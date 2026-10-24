"""Deterministic synthetic binary shapes; no downloaded training data."""

import random


def clean_image(seed, size=32):
    if size < 16:
        raise ValueError("Images must be at least 16 pixels wide")
    rng = random.Random(seed)
    image = [[0] * size for _ in range(size)]
    for shape in range(rng.randint(2, 6)):
        x0, y0 = rng.randrange(1, size - 6), rng.randrange(1, size - 6)
        x1, y1 = min(size - 2, x0 + rng.randint(3, size // 2)), min(size - 2, y0 + rng.randint(3, size // 2))
        value = 0 if shape > 1 and rng.random() < 0.2 else 1
        ellipse = rng.random() < 0.5
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if not ellipse or ((2*x-x0-x1)/(x1-x0))**2 + ((2*y-y0-y1)/(y1-y0))**2 <= 1:
                    image[y][x] = value
    return image


def noisy_image(clean, seed, probability=0.15):
    if not 0 <= probability <= 1:
        raise ValueError("Noise probability must be in [0, 1]")
    rng = random.Random(seed)
    return [[value ^ int(rng.random() < probability) for value in row] for row in clean]


def samples(count, seed=900000, size=32, probability=0.15):
    clean = [clean_image(seed + i, size) for i in range(count)]
    noisy = [noisy_image(image, seed + 1000000 + i, probability) for i, image in enumerate(clean)]
    return clean, noisy


def error_rate(predictions, targets):
    errors = sum(a != b for image, target in zip(predictions, targets)
                 for row, expected in zip(image, target) for a, b in zip(row, expected))
    count = sum(len(row) for image in targets for row in image)
    return errors / count


def majority(image):
    height, width = len(image), len(image[0])
    return [[int(sum(image[ny][nx] for ny in range(max(0, y-1), min(height, y+2))
                     for nx in range(max(0, x-1), min(width, x+2))) >= 5)
             for x in range(width)] for y in range(height)]
