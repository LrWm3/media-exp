"""Search six goals, retain scalar evidence, and validate real software decoders."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import codec_compute as media
from .lower import SynthesisError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("all", "h264", "hevc", "av1"), default="all")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "synthesis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"format": "codec-compute-synthesis-validation-1", "python": platform.python_version(),
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "hardware_validation": False, "cases": []}
    goals = (("min", media.minimum, 255), ("max", media.maximum, 255),
             ("and", media.bit_and, 1), ("or", media.bit_or, 1), ("xor", media.bit_xor, 1),
             ("absdiff", media.absdiff, 255))
    for target in ("h264", "hevc", "av1"):
        if args.target not in ("all", target):
            continue
        for name, function, maximum in goals:
            entry = {"goal": name, "target": target}
            try:
                program = media.compile(function, target=target, width=16, height=16,
                                        input_ranges={"a": (0, maximum), "b": (0, maximum)})
            except SynthesisError as error:
                entry.update({"status": error.result.status, "search": error.result.manifest(),
                              "backend_validation": None})
                report["cases"].append(entry)
                print(f"{target:4s} {name:3s}: {error.result.status} (bounded grammar only)")
                continue
            inputs = ({"a": [[x % 2 for x in range(16)] for _ in range(16)],
                       "b": [[(x // 2) % 2 for x in range(16)] for _ in range(16)]} if maximum == 1 else
                      {"a": [[(x * 19 + y * 71) % 256 for x in range(16)] for y in range(16)],
                       "b": [[(x * 73 + y * 13 + 29) % 256 for x in range(16)] for y in range(16)]})
            decoders = ["libdav1d", "libaom-av1"] if target == "av1" else ["ffmpeg"]
            for decoder in decoders:
                actual = program.verify(inputs, decoder=decoder)
            path = program.save(args.output / f"{name}-{target}{program.native.suffix}", inputs)
            manifest = program.manifest()
            entry.update({"status": "found", "search": program.synthesis_records[0]["search"],
                          "stream": path.name, "manifest": path.with_suffix(".json").name,
                          "costs": manifest["costs"], "prediction_sha256": manifest["prediction_sha256"],
                          "backend_validation": {"method": "software_decoder_against_semantic_interpreter",
                                                 "decoders": decoders, "cases": 1, "pixels_per_case": 256,
                                                 "output_sha256": hashlib.sha256(bytes(value for row in actual["output"]
                                                                                       for value in row)).hexdigest()}})
            report["cases"].append(entry)
            verified = entry["search"]["verification"]
            print(f"{target:4s} {name:3s}: {len(entry['search']['plan']['instructions'])} scalar instructions; "
                  f"{verified['combinations_checked']} exhaustive states; software decoder exact")
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
