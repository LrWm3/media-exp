#!/usr/bin/env python3
"""Compile symbolic Life, decode the saved H.264 stream, and capture its pixels."""

import argparse
import json
from pathlib import Path
import subprocess

from life_kernel import compile_life, generation_frames, verify_generations
from decoder_core.decode import decode
from life import seed_board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", choices=("glider", "blinker", "block", "r-pentomino", "random", "empty"), default="r-pentomino")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "out/life.h264")
    parser.add_argument("--decoder", choices=("ffmpeg", "openh264", "vaapi"), default="ffmpeg")
    parser.add_argument("--verify", action="store_true", help="Check every decoded generation against the independent Life oracle")
    parser.add_argument("--preview", action="store_true", help="Render a GIF from actual decoded generations; implies verification")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args()
    try:
        program = compile_life(args.width, args.height, args.generations,
                               optimize=not args.no_optimize)
        board = seed_board(args.pattern, args.width, args.height, args.random_seed)
        program.save(args.output, board)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}: {args.output.stat().st_size:,} bytes; {program.pictures} decoded pictures", flush=True)
    outputs = program.read(decode(args.output, args.decoder))
    if args.verify or args.preview:
        verify_generations(outputs, board, args.generations)
        print(f"PASS ({args.decoder}): {args.generations + 1} decoded generations match the independent Life oracle")
    args.output.with_suffix(".result.json").write_text(json.dumps({
        "decoder": args.decoder, "generations": args.generations,
        "width": args.width, "height": args.height,
        "output_frames": program.output_frames, "output": outputs["output"],
    }, indent=2) + "\n")
    if args.preview:
        preview = args.output.with_suffix(".gif")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                        "-s", f"{args.width}x{args.height}", "-framerate", "12", "-i", "pipe:0",
                        "-vf", "scale=512:-1:flags=neighbor", "-loop", "0", "-y", str(preview)],
                       input=b"".join(generation_frames(outputs, args.generations)),
                       check=True, timeout=120)
        print(f"Capture from decoded pixels: {preview}")


if __name__ == "__main__":
    main()
