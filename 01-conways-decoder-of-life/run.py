#!/usr/bin/env python3
"""Generate, verify and optionally preview a decoder-executed Life program."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

from life import seed_board
from program import compile_stream, FRAMES_PER_GENERATION
from verify import check, decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=("glider", "blinker", "block", "r-pentomino", "random", "empty"), default="r-pentomino")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("out/life.h264"))
    parser.add_argument("--verify", action="store_true", help="Compare decoded generations with a Python oracle")
    parser.add_argument("--decoder", choices=("ffmpeg", "vaapi", "openh264"), default="ffmpeg")
    parser.add_argument("--preview", action="store_true", help="Also save a GIF from decoded generation frames (implies verification)")
    args = parser.parse_args()
    if args.width < 16 or args.height < 16 or args.width % 16 or args.height % 16:
        parser.error("width and height must be positive multiples of 16")
    if args.generations < 0:
        parser.error("generations cannot be negative")
    board = seed_board(args.pattern, args.width, args.height, args.random_seed)
    stream, program = compile_stream(board, args.generations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(stream)
    manifest = {
        "width": args.width, "height": args.height, "guard": 16,
        "generations": args.generations, "frames_per_generation": FRAMES_PER_GENERATION,
        "seed": board,
        "pictures": [{"frame": 0, "label": "I_PCM initial state"}] +
                    [{"frame": i, **asdict(op)} for i, op in enumerate(program, 1)],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.output}: {len(stream):,} bytes; {len(program) + 1} decoded pictures")
    if args.verify or args.preview:
        raw = decode(args.output, args.decoder)
        frames = check(raw, board, args.generations)
        print(f"PASS ({args.decoder}): all {args.generations + 1} states match; every temporary has zero guard and neutral chroma")
        if args.preview:
            output = args.output.with_suffix(".gif")
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                            "-s", f"{args.width}x{args.height}", "-framerate", "12", "-i", "pipe:0",
                            "-vf", "scale=512:-1:flags=neighbor", "-loop", "0", "-y", str(output)],
                           input=b"".join(frames), check=True, timeout=120)
            print(f"Preview from decoded pixels: {output}")


if __name__ == "__main__":
    main()
