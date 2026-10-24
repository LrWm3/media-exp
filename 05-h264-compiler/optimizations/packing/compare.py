#!/usr/bin/env python3
"""Compare stable mixed packing with cost-guarded automatic scheduling.

All eight committed denoising inputs run at 2/3/4/8 tiles; committed Life and
the current maze graph run at 2/4 tiles. Every named output is checked against
the old graph reference and historical decoded outputs by the IR adapters.
The generated suite works with ../ir/validate.py and ../ir/compact.py.
No decoder timings are measured.
"""

import argparse
import importlib.util
from itertools import chain
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("packing_ir_adapters", ROOT.parent / "ir" / "compare.py")
IR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IR
SPEC.loader.exec_module(IR)
h264, write_json, sha256 = IR.h264, IR.write_json, IR.sha256
GROUPS = ("denoise", "life", "maze-current", "chain-leaves")
CAPACITIES = {"denoise": (2, 3, 4, 8), "life": (2, 4),
              "maze-current": (2, 4), "chain-leaves": (2, 3, 4, 8)}
GUARDED = ("pictures", "prediction_slices", "prediction_bytes", "output_readback_bytes")


def synthetic():
    def kernel(image):
        outputs = {f"leaf_{index}": h264.saturating_weighted(image, 1, 2*index+1)
                   for index in range(8)}
        value = image
        for index in range(8):
            value = h264.saturating_weighted(h264.shift(value, 1 if index % 2 else -1), 2, -63)
            outputs[f"stage_{index}"] = value
        outputs["output"] = value
        return outputs
    # Every stage is named, so the compiler must retain the complete chain.
    program = h264.compile(kernel, 16, 16, input_range=(0, 255))
    inputs = [[y*16+x for x in range(16)] for y in range(16)]
    yield IR.Comparison("chain-leaves", "chain-leaves", {"optimized": program}, inputs,
                        program.reference(inputs), "scheduling",
                        {"input_source": "all 256 byte values, row-major 16x16 ramp",
                         "independent_leaves": 8, "chain_stages": 8})


def comparisons(names, sources):
    saved = IR.saved_cases(set(names) - {"chain-leaves"}, sources)
    packed = {}
    for original in chain(saved, synthetic() if "chain-leaves" in names else ()):
        program = original.programs["optimized"]
        for tiles in CAPACITIES[original.group]:
            group = f"{original.group}-tiles{tiles}"
            if group not in packed:
                packed[group] = {variant: h264.pack(program, tiles=tiles, mixed_predictions=True,
                                                    scheduling=scheduling)
                                 for variant, scheduling in (("baseline", "stable"), ("optimized", "auto"))}
            name = f"{group}-{original.segment:02d}" if original.group == "denoise" else group
            yield IR.Comparison(name, group, packed[group], original.inputs, original.expected,
                                "scheduling", {**original.source, "tiles": tiles}, original.segment)


def geometry_probes(sources):
    """Record larger saved shapes separately; never count these as decoded cases."""
    result = []
    for saved in IR.SAVED.selected_cases(["mnist", "motion"]):
        path = IR.SAVED.REPOSITORY / saved.manifest
        payload = path.read_bytes()
        sources[saved.manifest] = sha256(payload)
        manifest = json.loads(payload)
        original = IR.SAVED.compile_saved_graph(manifest)
        for tiles in (2, 4):
            record = {"graph": saved.name, "tiles": tiles, "source_manifest": saved.manifest,
                      "width": original.width, "height": original.height,
                      "decoded": False}
            try:
                packed = h264.pack(original, tiles=tiles, scheduling="auto")
            except h264.CompileError as error:
                record.update(status="unsupported", reason=str(error))
            else:
                record.update(status="packable_not_decoded", costs=packed.costs)
            result.append(record)
    return result


def assert_costs(before, after):
    for key in ("coded_width", "coded_height", "input_pictures"):
        if before[key] != after[key]:
            raise AssertionError("Scheduling changed geometry or inputs: " + key)
    for key in GUARDED:
        if after[key] > before[key]:
            raise AssertionError(f"Automatic scheduling regressed {key}: {before[key]} -> {after[key]}")


def prepare(output, names=GROUPS, local_backends=("ffmpeg", "openh264", "vaapi"), probe_geometry=True):
    if not names or set(names) - set(GROUPS):
        raise ValueError("Unknown or empty case selection")
    if set(local_backends) - {"ffmpeg", "openh264", "vaapi", "nvdec"}:
        raise ValueError("Unknown decoder")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    backends = tuple(dict.fromkeys(("ffmpeg", *local_backends)))
    status = {name: {"status": "pending" if IR.available(name) else "unavailable", "streams": 0}
              for name in backends}
    if status["ffmpeg"]["status"] == "unavailable":
        raise RuntimeError("FFmpeg required for saved inputs and complete I420 references")
    source_paths = [*sorted((IR.SAVED.COMPILER / "h264").glob("*.py")),
                    IR.SAVED.REPOSITORY / "decoder_core" / "bitstream.py",
                    IR.SAVED.REPOSITORY / "decoder_core" / "decode.py", ROOT.parent / "prepare.py",
                    ROOT.parent / "ir" / "compare.py", ROOT.parent / "ir" / "validate.py",
                    ROOT / "compare.py"]
    source_hashes = {str(path.relative_to(IR.SAVED.REPOSITORY)): sha256(path.read_bytes())
                     for path in source_paths}
    sources, records, cases, representatives = {}, [], [], {}
    for comparison in comparisons(names, sources):
        entries = {variant: IR.save_variant(output, comparison, variant, backends, status)
                   for variant in IR.VARIANTS}
        for variant, entry in entries.items():
            program = comparison.programs[variant]
            entry["costs"]["output_readback_bytes"] = len(program.output_picture_indices)*entry["frame_bytes"]
        assert_costs(entries["baseline"]["costs"], entries["optimized"]["costs"])
        record = {"name": comparison.name, "group": comparison.group, "segment": comparison.segment,
                  "source": comparison.source, "named_outputs_equal": True,
                  "old_graph_reference_equal": True,
                  "historical_decoded_outputs_equal": True if comparison.source["input_source"] == "committed decoded PCM" else None,
                  "cost_guard_pass": True, "named_outputs": len(comparison.expected),
                  "named_output_pixels": sum(len(row) for board in comparison.expected.values() for row in board),
                  **{variant: {key: entry[key] for key in ("costs", "compiler_stats", "stream_sha256",
                                                         "expected_sha256", "manifest_sha256")}
                     for variant, entry in entries.items()}}
        records.append(record)
        cases.extend(entries.values())
        if comparison.segment == 0:
            representatives[comparison.group] = {key: entries["optimized"][key] for key in
                                                ("width", "height", "coded_width", "coded_height", "frame_bytes")}
            representatives[comparison.group].update(entries)
        print(f"PASS {comparison.name}: {entries['baseline']['pictures']} -> "
              f"{entries['optimized']['pictures']} pictures; {record['named_outputs']} named outputs", flush=True)
    probes = geometry_probes(sources) if probe_geometry else []
    for item in status.values():
        if item["status"] != "unavailable":
            item["status"] = "pass"
    for path in source_paths:
        if source_hashes[str(path.relative_to(IR.SAVED.REPOSITORY))] != sha256(path.read_bytes()):
            raise AssertionError("Source changed during comparison; rerun with stable sources: " + str(path))
    report = {"format": "h264-packing-schedule-comparison-0.1", "status": "pass",
              "baseline": "scheduling=stable; existing FIFO mixed P/B scheduler",
              "optimized": "scheduling=auto; critical-path candidate with componentwise cost guard",
              "unchanged": "same compiled graph, input pixels, tiles, mixed_predictions=True, and all named outputs",
              "guarded_costs": list(GUARDED), "timings": "not measured",
              "oracle": "old compiler reference plus historical decoded named outputs; synthetic uses original graph reference",
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=IR.SAVED.REPOSITORY, text=True).strip(),
              "python": platform.python_version(),
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "source_sha256": source_hashes, "historical_source_sha256": sources,
              "local_backends": status, "comparison_count": len(records), "stream_count": len(cases),
              "named_outputs": sum(item["named_outputs"] for item in records),
              "named_output_pixels": sum(item["named_output_pixels"] for item in records),
              "comparisons": records, "geometry_probes": probes,
              "cost_definitions": {"pictures": "Full stream, including PCM inputs and normalization",
                  "prediction_slices": "VCL NAL count minus one PCM slice per input",
                  "prediction_bytes": "All stream bytes after the last PCM input NAL",
                  "output_readback_bytes": "Full I420 bytes for unique pictures containing named outputs",
                  "full_readback_bytes": "Full I420 bytes for every coded picture"}}
    write_json(output / "comparison.json", report)
    write_json(output / "suite.json", {"format": "h264-ir-suite-0.1",
        "repository_commit": report["repository_commit"],
        "comparison_sha256": sha256((output / "comparison.json").read_bytes()),
        "case_count": len(cases), "representatives": representatives, "cases": cases})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--cases", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--local-backends", nargs="+", choices=("ffmpeg", "openh264", "vaapi", "nvdec"),
                        default=["ffmpeg", "openh264", "vaapi"])
    parser.add_argument("--skip-geometry-probes", action="store_true")
    args = parser.parse_args()
    report = prepare(args.output, args.cases, args.local_backends, not args.skip_geometry_probes)
    print(json.dumps({key: report[key] for key in ("comparison_count", "stream_count", "named_outputs",
                                                  "named_output_pixels", "local_backends")}, indent=2))
