#!/usr/bin/env python3
"""Compile and capture a two-neighbor binary OR kernel."""

import argparse
from pathlib import Path
import subprocess

import h264
from decoder_core.decode import decode


@h264.kernel
def neighbor_or(image):
    right = h264.shift(image, dx=1, dy=0)
    below = h264.shift(image, dx=0, dy=1)
    return h264.threshold(h264.add(right, below), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("out/shifted-or.h264"))
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    image = [[0]*32 for _ in range(32)]
    for x, y in ((4, 4), (12, 12), (12, 13), (13, 12), (20, 20), (21, 21), (22, 22), (0, 28), (31, 4)):
        image[y][x] = 1
    program = h264.compile(neighbor_or, width=32, height=32)
    program.save(args.output, image)
    output = program.read(decode(args.output, args.decoder))["output"]
    if output != program.reference(image)["output"]:
        raise AssertionError("Decoder differs from graph")
    print(f"PASS ({args.decoder}): exact shifted OR; {program.pictures} pictures; {len(program.tail)} prediction bytes")
    print(f"Stream: {args.output}; manifest: {args.output.with_suffix('.json')}")
    if args.preview:
        width, height, scale = 720, 430, 10
        canvas = bytearray((11, 18, 32)*(width*height))
        for pixels, left in ((image, 24), (output, 376)):
            for y, row in enumerate(pixels):
                data = b"".join(bytes((225, 235, 249) if v else (23, 34, 51))*scale for v in row)
                for dy in range(scale):
                    start = ((90+y*scale+dy)*width+left)*3
                    canvas[start:start+len(data)] = data
        filters = ",".join(f"drawtext=font='DejaVu Sans':text='{text}':expansion=none:x={x}:y={y}:fontsize={size}:fontcolor=white"
                           for x, y, size, text in [(24, 18, 23, "PYTHON -> H.264 / SHIFTED OR"),
                                                   (24, 58, 16, "INPUT"), (376, 58, 16, f"DECODED OUTPUT / {args.decoder}")])
        subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                        "-i", "pipe:0", "-vf", filters, "-frames:v", "1", "-y", str(args.output.with_suffix('.png'))],
                       input=canvas, check=True, timeout=60)


if __name__ == "__main__":
    main()
