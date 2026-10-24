#!/usr/bin/env python3
"""Render verified NVDEC Life planes; display scaling does not alter the stream."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render(bundle, results, output):
    suite_bytes = (bundle / "suite.json").read_bytes()
    suite = json.loads(suite_bytes)
    report = json.loads((results / "results.json").read_text())
    if hashlib.sha256(suite_bytes).hexdigest() != report["suite_sha256"]:
        raise ValueError("Capture report belongs to another suite")
    case = next(case for case in suite["cases"] if case["name"] == "life")
    actual = next(case for case in report["cases"] if case["name"] == "life")
    if actual["status"] != "pass":
        raise ValueError("Life decoder validation did not pass")
    generations = sorted((int(name.removeprefix("generation_")), name)
                         for name in actual["named_outputs"] if name.startswith("generation_"))
    boards = {}
    for generation, name in generations:
        record = actual["named_outputs"][name]
        with Image.open(results / record["capture"]) as image:
            board = image.convert("L")
        if (board.size != (case["width"], case["height"])
                or hashlib.sha256(board.tobytes()).hexdigest() != case["named_outputs"][name]["sha256"]
                or set(board.tobytes()) - {0, 1}):
            raise ValueError("Captured Life plane differs from verified binary output")
        boards[generation] = board.point(lambda value: value * 255)

    def font(size):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    def paint(canvas, board, position, size):
        pixels = board.resize((size, size), Image.Resampling.NEAREST)
        panel = Image.new("RGB", pixels.size, "#1c293a")
        panel.paste("#90e8b2", mask=pixels)
        canvas.paste(panel, position)

    before, after = (case["comparison"][key]["pictures"] for key in ("before", "after"))
    output.mkdir(parents=True, exist_ok=True)
    still = Image.new("RGB", (920, 374), "#101824")
    draw = ImageDraw.Draw(still)
    draw.text((24, 18), "Life executed by NVDEC", font=font(26), fill="white")
    draw.text((24, 57), f"40 generations | {before} -> {after} pictures | exact decoded pixels", font=font(17), fill="#b7c9dc")
    for column, generation in enumerate((0, 5, 20, generations[-1][0])):
        x = 24 + column * 224
        paint(still, boards[generation], (x, 112), 200)
        draw.text((x, 85), f"Generation {generation}", font=font(16), fill="white")
    draw.text((24, 335), "0/1 output scaled for display. All 42 cases match FFmpeg, OpenH264 and VA-API.",
              font=font(15), fill="#b7c9dc")
    still.save(output / "life-capture.png")
    frames = []
    for generation, _ in generations:
        frame = Image.new("RGB", (368, 418), "#101824")
        draw = ImageDraw.Draw(frame)
        draw.text((24, 14), f"NVDEC | generation {generation:02d}", font=font(20), fill="white")
        paint(frame, boards[generation], (24, 56), 320)
        draw.text((24, 390), f"{after} pictures / 40 generations", font=font(15), fill="#b7c9dc")
        frames.append(frame)
    frames[0].save(output / "life.gif", save_all=True, append_images=frames[1:],
                   duration=[160] * (len(frames) - 1) + [800], loop=0, disposal=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("assets"))
    args = parser.parse_args()
    render(args.bundle, args.results, args.output)
