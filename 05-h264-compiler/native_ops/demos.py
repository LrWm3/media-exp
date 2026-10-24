#!/usr/bin/env python3
"""Compile six native decoder demonstrations and capture actual decoded pixels."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h264
from decoder_core.decode import available, decode
from common import BACKENDS, ROOT, contact_sheet, digest, fixture, provenance, save_plane, tile_fixture


def cases(width=64, height=64):
    images = fixture(width, height)
    cols, rows = width//16, height//16
    # A gather offset identifies source minus destination, in quarter pixels.
    permutation = []
    warp = []
    for by in range(rows):
        for bx in range(cols):
            destination = by*cols+bx
            source = cols*rows-1-destination
            permutation.append(((source % cols-bx)*64, (source//cols-by)*64))
            warp.append(((by-rows//2)*12, (bx-cols//2)*3))

    def add(a, b):
        return h264.saturating_add(a, b)

    def affine(image):
        return {"exposure_plus_48": h264.saturating_weighted(image, bias=48),
                "contrast_2x_minus_64": h264.saturating_weighted(image, w=2, bias=-64)}

    def tiles(image):
        return h264.gather(image, offsets=permutation)

    def shear(image):
        return h264.gather(image, offsets=warp)

    def fractional(image):
        return {"integer_x_plus_1": h264.shift(image, dx=1),
                "half_x": h264.sample(image, qx=2, qy=0),
                "quarter_x_half_y": h264.sample(image, qx=1, qy=2)}

    def chain(a, b):
        mixed = h264.saturating_add(a, b)
        contrast = h264.saturating_weighted(mixed, w=2, bias=-96)
        translated = h264.sample(contrast, qx=1, qy=2)
        value = translated
        for index in range(40):
            value = h264.saturating_weighted(value, bias=2 if index % 2 == 0 else -1)
        difference = h264.saturating_combine(value, a, wa=1, wb=-1, bias=96)
        output = h264.saturating_combine(difference, b, bias=-32)
        return {"mixed": mixed, "contrast": contrast, "fractional": translated,
                "after_40_stages": value, "difference": difference, "output": output}

    common = {"width": width, "height": height}
    named = {**common, "input_ranges": {"a": (0, 255), "b": (0, 255)}}
    single = {**common, "input_range": (0, 255)}
    return [
        ("saturating-add", "Two input u8 saturating addition", add, named, images),
        ("affine", "Exposure and contrast", affine, single, images["a"]),
        ("tile-permutation", "16 x 16 block permutation", tiles, single, tile_fixture(width, height)),
        ("block-warp", "Blockwise shear / fractional warp", shear, single, images["a"]),
        ("fractional-translation", "Native fractional luma interpolation", fractional, single, images["a"]),
        ("resident-stages", "Chained stages and live-reference refresh", chain, named, images),
    ]


def run(output, width, height, backends):
    output.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "shape": [width, height], "provenance": provenance(), "cases": []}
    for name, title, function, options, inputs in cases(width, height):
        program = h264.compile(function, **options)
        stream = program.save(output/"streams"/f"{name}.h264", inputs)
        expected = program.reference(inputs)
        input_boards = inputs if isinstance(inputs, dict) else {"input": inputs}
        for label, board in input_boards.items():
            save_plane(board, output/"captures"/name/f"input-{label}.png")
        results, decoded = {}, {}
        for backend in backends:
            if not available(backend):
                results[backend] = {"status": "unavailable"}
                continue
            actual = program.read(decode(stream, backend))
            if actual != expected:
                raise AssertionError(f"{name}: {backend} differs from independent integer reference")
            decoded[backend] = actual
            for label, board in actual.items():
                save_plane(board, output/"captures"/name/f"{backend}-{label}.png")
            results[backend] = {"status": "pass", "outputs": len(actual),
                                "checked_output_pixels": width*height*len(actual),
                                "output_sha256": {label: digest(board) for label, board in actual.items()}}
            print(f"PASS {name}: {backend}; {len(actual)} exact outputs", flush=True)
        if not decoded:
            raise RuntimeError("No requested decoder available")
        capture_backend = "vaapi" if "vaapi" in decoded else next(iter(decoded))
        contact_sheet(title,
                      f"Actual {capture_backend} output | {program.input_count} PCM inputs + "
                      f"{len(program.operations)} prediction pictures | {len(program.refreshes)} refresh copies",
                      [(f"HOST / {key}", value) for key, value in input_boards.items()]
                      +[(f"DECODED / {key}", value) for key, value in decoded[capture_backend].items()],
                      output/"assets"/f"{name}.png")
        report["cases"].append({"name": name, "title": title, "pictures": program.pictures,
                                "inputs": program.input_count, "prediction_pictures": len(program.operations),
                                "refresh_copies": len(program.refreshes), "output_frames": program.output_frames,
                                "input_bytes": len(program.encode_inputs(inputs)),
                                "prediction_bytes": len(program.tail), "stream_bytes": stream.stat().st_size,
                                "capture_backend": capture_backend, "decoders": results})
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--decoders", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    args = parser.parse_args()
    run(args.output, args.width, args.height, args.decoders)


if __name__ == "__main__":
    main()
