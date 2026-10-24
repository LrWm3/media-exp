#!/usr/bin/env python3
"""Compare semantic IR and mixed P/B packing against their disabled baselines.

Reuse committed Life, Maze, denoising, MNIST and motion PCM inputs; no training
or dataset downloads. Every original segment and named output is checked against
the old graph reference and every available requested decoder. Costs are static;
this command does not report timings. The bundle supports validate.py and the
existing ../benchmark.py (which measures its first-segment representatives).
"""

import argparse
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ir_saved_graphs", ROOT.parent / "prepare.py")
SAVED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAVED
SPEC.loader.exec_module(SAVED)
h264, decode, available = SAVED.h264, SAVED.decode, SAVED.available
sha256, write_json = SAVED.sha256, SAVED.write_json

SAVED_GROUPS = ("life", "maze", "maze-current", "denoise", "mnist", "motion")
SYNTHETIC = ("parity40", "absorption", "equivalent-or", "mixed-pb")
PACKED = ("denoise-packed4", "denoise-packed8")
VARIANTS = ("baseline", "optimized")


@dataclass
class Comparison:
    name: str
    group: str
    programs: dict
    inputs: object
    expected: dict
    change: str
    source: dict
    segment: int = 0


def synthetic_cases(names):
    def parity(a, b):
        result = a
        for _ in range(40):
            result = h264.bit_or(h264.bit_and(result, h264.bit_not(b)),
                                h264.bit_and(h264.bit_not(result), b))
        return {"output": result, "a": a, "b": b}

    def absorption(a, b):
        return {"or_absorption": h264.bit_or(a, h264.bit_and(a, b)),
                "and_absorption": h264.bit_and(a, h264.bit_or(a, b)), "a": a}

    def equivalent(a, b):
        return {"or": h264.bit_or(a, b),
                "demorgan": h264.bit_not(h264.bit_and(h264.bit_not(a), h264.bit_not(b)))}

    def mixed(a, b):
        unary = h264.saturating_weighted(a, -1, 127)
        binary = h264.saturating_add(a, b)
        return {"unary": unary, "binary": binary,
                "combined": h264.saturating_add(unary, binary)}

    for name, kernel, count in (("parity40", parity, 2), ("absorption", absorption, 2),
                                ("equivalent-or", equivalent, 2), ("mixed-pb", mixed, 2)):
        if name not in names:
            continue
        labels = "abc"[:count]
        high = 255 if name == "mixed-pb" else 1
        options = {"width": 16, "height": 16,
                   "input_ranges": {label: (0, high) for label in labels}}
        inputs = {label: [[((y*16+x)*(2*index+1)+17*index) % 256 if high == 255
                            else ((y*16+x) >> index) & 1 for x in range(16)] for y in range(16)]
                  for index, label in enumerate(labels)}
        before = h264.compile(kernel, **options, semantic_ir=False)
        after = h264.compile(kernel, **options, semantic_ir=True)
        expected = before.reference(inputs)
        if name == "mixed-pb":
            before, after = (h264.pack(after, tiles=2, mixed_predictions=mixing)
                             for mixing in (False, True))
        yield Comparison(name, name, dict(zip(VARIANTS, (before, after))), inputs, expected,
                         "mixed_predictions" if name == "mixed-pb" else "semantic_ir",
                         {"input_source": "deterministic exhaustive binary tuples or u8 ramps",
                          "logical_rounds": 40 if name == "parity40" else None})


def saved_cases(names, sources):
    groups = [name for name in SAVED_GROUPS if name in names and name != "maze-current"]
    if any(name in names for name in PACKED) and "denoise" not in groups:
        groups.append("denoise")
    selected = list(SAVED.selected_cases(groups)) if groups else []
    if "maze-current" in names:
        base = "02-maze-flood-fill-decoder-solver/compiled/optimizations/assets/maze"
        selected.append(SAVED.SavedCase("maze-current", "maze-current", base + ".h264", base + ".json"))
    for saved in selected:
        stream_path = SAVED.REPOSITORY / saved.stream
        manifest_path = SAVED.REPOSITORY / saved.manifest
        stream_bytes, manifest_bytes = stream_path.read_bytes(), manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("stream_sha256") and manifest["stream_sha256"] != sha256(stream_bytes):
            raise AssertionError(saved.name + ": historical stream hash mismatch")
        sources[saved.stream], sources[saved.manifest] = sha256(stream_bytes), sha256(manifest_bytes)
        segments = SAVED.split_segments(stream_bytes)
        if len(segments) != saved.segments:
            raise AssertionError(saved.name + ": historical segment count changed")
        raw = decode(stream_path, "ffmpeg")
        chunk_bytes = manifest["pictures"]*(manifest["width"]+32)*(manifest["height"]+32)*3//2
        if len(raw) != len(segments)*chunk_bytes:
            raise AssertionError(saved.name + ": historical decoded picture count changed")
        before = SAVED.compile_saved_graph(manifest, semantic_ir=False)
        after = SAVED.compile_saved_graph(manifest, semantic_ir=True)
        versions = [(saved.name, "semantic_ir", before, after)] if saved.group in names else []
        if saved.name == "denoise":
            versions += [(f"denoise-packed{tiles}", "mixed_predictions",
                          h264.pack(after, tiles=tiles, mixed_predictions=False),
                          h264.pack(after, tiles=tiles, mixed_predictions=True))
                         for tiles in (4, 8) if f"denoise-packed{tiles}" in names]
        for index in range(len(segments)):
            original = raw[index*chunk_bytes:(index+1)*chunk_bytes]
            historical = SAVED.TOOLS.verify_saved_graph(original, manifest)
            inputs = SAVED.extract_inputs(original, manifest, before.named_inputs)
            expected = before.reference(inputs)
            if expected != historical:
                raise AssertionError(saved.name + ": old compiler differs from saved graph semantics")
            for group, change, baseline, optimized in versions:
                name = group if len(segments) == 1 else f"{group}-{index:02d}"
                yield Comparison(name, group, dict(zip(VARIANTS, (baseline, optimized))), inputs,
                                 expected, change, {"stream": saved.stream, "manifest": saved.manifest,
                                 "input_source": "committed decoded PCM",
                                 "historical_segment_sha256": sha256(segments[index]),
                                 "items": saved.items_per_segment}, index)
        if stream_path.read_bytes() != stream_bytes or manifest_path.read_bytes() != manifest_bytes:
            raise AssertionError(saved.name + ": historical source changed during comparison")


def costs(program, stream):
    units = SAVED.annexb_units(stream)
    vcl = [unit for unit in units if unit[2] in (1, 5)]
    prediction_start = vcl[program.input_count-1][1]
    pixels = program.writer.width*program.writer.height
    return {"pictures": program.pictures, "input_pictures": program.input_count,
            "prediction_pictures": program.pictures-program.input_count,
            "prediction_slices": len(vcl)-program.input_count,
            "stream_bytes": len(stream), "prediction_bytes": len(stream)-prediction_start,
            "peak_live_values": program.stats["peak_live_values"],
            "coded_width": program.writer.width, "coded_height": program.writer.height,
            "coded_luma_samples": pixels*program.pictures,
            "full_readback_bytes": pixels*program.pictures*3//2}


def save_variant(root, comparison, variant, backends, status):
    program = comparison.programs[variant]
    expected = comparison.expected
    if program.reference(comparison.inputs) != expected:
        raise AssertionError(f"{comparison.name}/{variant}: graph differs from baseline reference")
    stem = comparison.name + "-" + variant
    stream = program.save(root / "streams" / (stem + ".h264"), comparison.inputs)
    raw = decode(stream, "ffmpeg")
    if program.read(raw) != expected:
        raise AssertionError(stem + ": FFmpeg differs from baseline reference")
    for backend in backends:
        if status[backend]["status"] == "unavailable":
            continue
        actual = raw if backend == "ffmpeg" else decode(stream, backend)
        if program.read(actual) != expected:
            raise AssertionError(stem + ": " + backend + " named outputs differ from baseline reference")
        if actual != raw:
            raise AssertionError(stem + ": " + backend + " full I420 differs from FFmpeg")
        status[backend]["streams"] += 1
    expected_path = root / "expected" / (stem + ".i420")
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_bytes(raw)
    outputs = {}
    for index, (name, board) in enumerate(expected.items()):
        pixels = SAVED.TOOLS.plane_bytes(board)
        path = root / "named_outputs" / comparison.name / f"{index:03d}.u8"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pixels)
        region = getattr(program, "output_regions", {}).get(name, {"x": 0, "y": 0})
        outputs[name] = {"path": str(path.relative_to(root)), "sha256": sha256(pixels),
                         "bytes": len(pixels), "frame": program.output_frames[name],
                         "x": region["x"]+16, "y": region["y"]+16}
    entry = {"name": stem, "variant": variant, "group": comparison.group,
             "stream": str(stream.relative_to(root)), "stream_sha256": sha256(stream.read_bytes()),
             "manifest": str(stream.with_suffix(".json").relative_to(root)),
             "manifest_sha256": sha256(stream.with_suffix(".json").read_bytes()),
             "expected": str(expected_path.relative_to(root)), "expected_sha256": sha256(raw),
             "expected_bytes": len(raw), "pictures": program.pictures,
             "width": program.width, "height": program.height,
             "coded_width": program.writer.width, "coded_height": program.writer.height,
             "frame_bytes": program.writer.width*program.writer.height*3//2,
             "named_outputs": outputs, "costs": costs(program, stream.read_bytes()),
             "compiler_stats": program.stats}
    return entry


def prepare(output, names=None, local_backends=("ffmpeg", "openh264", "vaapi")):
    selected = set(names or (*SAVED_GROUPS, *SYNTHETIC, *PACKED))
    unknown = selected - set((*SAVED_GROUPS, *SYNTHETIC, *PACKED))
    if unknown:
        raise ValueError("Unknown cases: " + ", ".join(sorted(unknown)))
    if set(local_backends) - {"ffmpeg", "openh264", "vaapi", "nvdec"}:
        raise ValueError("Unknown decoder")
    backends = tuple(dict.fromkeys(("ffmpeg", *local_backends)))
    status = {backend: {"status": "pending" if available(backend) else "unavailable", "streams": 0}
              for backend in backends}
    if status["ffmpeg"]["status"] == "unavailable":
        raise RuntimeError("FFmpeg required for saved inputs and full I420 reference")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_paths = [*sorted((SAVED.COMPILER / "h264").glob("*.py")),
                    SAVED.REPOSITORY / "decoder_core" / "bitstream.py",
                    SAVED.REPOSITORY / "decoder_core" / "decode.py", ROOT.parent / "prepare.py",
                    ROOT / "compare.py", ROOT / "validate.py"]
    source_hashes = {str(path.relative_to(SAVED.REPOSITORY)): sha256(path.read_bytes())
                     for path in source_paths}
    sources, comparisons, cases, representatives = {}, [], [], {}
    from itertools import chain
    for comparison in chain(synthetic_cases(selected), saved_cases(selected, sources)):
        entries = {variant: save_variant(output, comparison, variant, backends, status)
                   for variant in VARIANTS}
        cases.extend(entries.values())
        record = {"name": comparison.name, "group": comparison.group, "change": comparison.change,
                  "segment": comparison.segment, "source": comparison.source,
                  "named_outputs_equal": True, "baseline_reference_equal": True,
                  "named_outputs": len(comparison.expected),
                  "named_output_pixels": sum(len(row) for board in comparison.expected.values() for row in board),
                  **{variant: {key: entry[key] for key in ("costs", "compiler_stats", "stream_sha256",
                                                         "expected_sha256", "manifest_sha256")}
                     for variant, entry in entries.items()}}
        comparisons.append(record)
        if comparison.segment == 0:
            sample = entries["optimized"]
            representatives[comparison.group] = {key: sample[key] for key in
                                                ("width", "height", "coded_width", "coded_height", "frame_bytes")}
            representatives[comparison.group].update(entries)
        print("PASS {}: {} -> {} pictures; {} named outputs".format(
            comparison.name, entries["baseline"]["pictures"], entries["optimized"]["pictures"],
            len(comparison.expected)), flush=True)
    for item in status.values():
        if item["status"] != "unavailable":
            item["status"] = "pass"
    for path in source_paths:
        if source_hashes[str(path.relative_to(SAVED.REPOSITORY))] != sha256(path.read_bytes()):
            raise AssertionError("Source changed during comparison; rerun with stable sources: " + str(path))
    report = {"format": "h264-ir-comparison-0.1", "status": "pass",
              "baseline": "semantic_ir=False for IR cases; mixed_predictions=False for packing cases",
              "optimized": "semantic_ir=True for IR cases; mixed_predictions=True for packing cases",
              "packing_ir": "semantic_ir=True in both packing variants",
              "unchanged": "optimize=True, exact_search=True, reference_mode=retained, slice_mode=row",
              "oracle": "semantic_ir=False baseline.reference; saved graphs additionally match historical outputs",
              "timings": "not measured",
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=SAVED.REPOSITORY, text=True).strip(),
              "python": platform.python_version(),
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "source_sha256": source_hashes,
              "historical_source_sha256": sources, "local_backends": status,
              "comparison_count": len(comparisons), "stream_count": len(cases),
              "named_outputs": sum(item["named_outputs"] for item in comparisons),
              "named_output_pixels": sum(item["named_output_pixels"] for item in comparisons),
              "cost_definitions": {"pictures": "Includes PCM inputs, normalization and all prediction pictures",
                  "prediction_slices": "VCL NAL count minus one PCM slice per input",
                  "prediction_bytes": "All bytes after the last PCM input NAL",
                  "coded_luma_samples": "Full coded width * coded height * pictures",
                  "full_readback_bytes": "Full I420 readback including guards, unused tiles and chroma"},
              "comparisons": comparisons}
    write_json(output / "comparison.json", report)
    suite = {"format": "h264-ir-suite-0.1", "repository_commit": report["repository_commit"],
             "comparison_sha256": sha256((output / "comparison.json").read_bytes()),
             "case_count": len(cases), "representatives": representatives, "cases": cases}
    write_json(output / "suite.json", suite)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--cases", nargs="+", choices=(*SAVED_GROUPS, *SYNTHETIC, *PACKED))
    parser.add_argument("--local-backends", nargs="+", choices=("ffmpeg", "openh264", "vaapi", "nvdec"),
                        default=["ffmpeg", "openh264", "vaapi"])
    args = parser.parse_args()
    report = prepare(args.output, args.cases, args.local_backends)
    print(json.dumps({key: report[key] for key in ("comparison_count", "stream_count", "named_outputs",
                                                  "named_output_pixels", "local_backends")}, indent=2))


if __name__ == "__main__":
    main()
