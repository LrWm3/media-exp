"""Generate and verify reviewable streams for the three codec targets."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import codec_compute as media


def mean_chain(a, b):
    mean = media.average(a, b)
    return {"output": media.average(mean, a), "mean": mean, "original": b}


def signed_chain(a, b):
    total = media.add_sat(a, b)
    difference = media.sub_sat(total, b)
    return {"output": media.threshold(difference, 128), "sum": total, "difference": difference}


def translated_mean(a, b):
    mean = media.average(a, b)
    moved = media.shift(mean, dx=-3, dy=5, border="clamp")
    return {"output": media.average(moved, a), "mean": mean, "translated": moved}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    parser.add_argument("--target", choices=("all", "h264", "hevc", "av1"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = (("mean-chain", mean_chain, ("h264", "hevc", "av1")),
             ("signed-chain", signed_chain, ("h264", "hevc")),
             ("translated-mean", translated_mean, ("av1",)))
    inputs = {"a": [[(x * 19 + y * 71) % 256 for x in range(32)] for y in range(32)],
              "b": [[(x * 73 + y * 13 + 29) % 256 for x in range(32)] for y in range(32)]}
    report = {"format": "codec-compute-validation-1", "python": platform.python_version(),
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "hardware_validation": False, "cases": []}
    for name, function, targets in cases:
        graph = media.trace(function, width=32, height=32, input_ranges={"a": (0, 255), "b": (0, 255)})
        for target in targets:
            if args.target not in ("all", target):
                continue
            program = media.compile_graph(graph, target=target)
            result = program.verify(inputs)
            decoders = ["libdav1d" if target == "av1" else "ffmpeg"]
            if target == "av1":
                program.verify(inputs, decoder="libaom-av1")
                decoders.append("libaom-av1")
            path = program.save(args.output / f"{name}-{target}{program.native.suffix}", inputs)
            manifest = program.manifest()
            entry = {"case": name, "target": target, "verified": True,
                     "decoders": decoders, "stream": path.name, "stream_bytes": path.stat().st_size,
                     "prediction_sha256": manifest["prediction_sha256"],
                     "costs": manifest["costs"],
                     "outputs_sha256": {key: hashlib.sha256(bytes(value for row in plane for value in row)).hexdigest()
                                        for key, plane in result.items()}}
            report["cases"].append(entry)
            print(f"{name:18s} {target:4s} exact; {entry['costs']['total_pictures']} pictures; "
                  f"{entry['costs']['prediction_bytes']} prediction bytes")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
