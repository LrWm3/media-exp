#!/usr/bin/env python3
"""Recompile committed graphs and prepare a portable optimization comparison.

The historical streams supply PCM inputs and verified named outputs. Their saved
graphs independently evaluate every named output, then the current compiler
replays exactly those graph semantics. No models, fixtures, or historical assets
are regenerated. Each repeated graph execution becomes its own bundle case.
The baseline can be historical streams or the current compiler with only exact
instruction search disabled (--baseline no-exact-search).

Example: python3 prepare.py --output out/bundle --cases native motion
The resulting suite.json works with ../native_ops/nvdec/run.py. FFmpeg is always
required; available requested VAAPI/OpenH264 decoders must match every I420 byte.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
COMPILER = ROOT.parent
REPOSITORY = COMPILER.parent
sys.path[:0] = [str(COMPILER), str(REPOSITORY)]

import h264
from decoder_core.decode import available, decode
from h264.graph import node


def load_bundle_tools():
    path = COMPILER / "native_ops" / "nvdec" / "prepare.py"
    spec = importlib.util.spec_from_file_location("optimization_bundle_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = load_bundle_tools()
Bundle = TOOLS.Bundle
sha256 = TOOLS.sha256
write_json = TOOLS.write_json


@dataclass(frozen=True)
class SavedCase:
    name: str
    group: str
    stream: str
    manifest: str
    segments: int = 1
    items_per_segment: int = 1


SAVED_CASES = (
    SavedCase("life", "life", "01-conways-decoder-of-life/compiled/assets/life.h264",
              "01-conways-decoder-of-life/compiled/assets/life.json"),
    SavedCase("maze", "maze", "02-maze-flood-fill-decoder-solver/compiled/assets/maze.h264",
              "02-maze-flood-fill-decoder-solver/compiled/assets/maze.json"),
    SavedCase("denoise", "denoise", "03-denoising-decoder/compiled/assets/denoising.h264",
              "03-denoising-decoder/compiled/assets/program.json", 8),
    SavedCase("mnist", "mnist", "04-binary-mnist-classification-decoder/compiled/assets/mnist.h264",
              "04-binary-mnist-classification-decoder/compiled/assets/program.json", 2, 8),
    *(SavedCase("native-" + name, "native", "05-h264-compiler/native_ops/streams/" + name + ".h264",
                "05-h264-compiler/native_ops/streams/" + name + ".json")
      for name in ("saturating-add", "affine", "tile-permutation", "block-warp",
                   "fractional-translation", "resident-stages")),
    SavedCase("motion", "motion", "06-motion-mask-decoder/assets/motion.h264",
              "06-motion-mask-decoder/assets/program.json", 24),
)


def selected_cases(names=None):
    if names is None:
        return SAVED_CASES
    names = set(names)
    known = {item.name for item in SAVED_CASES} | {item.group for item in SAVED_CASES}
    if not names or names - known:
        raise ValueError("Unknown or empty case selection: " + ", ".join(sorted(names - known)))
    return tuple(item for item in SAVED_CASES if item.name in names or item.group in names)


def tuple_params(value):
    """JSON arrays nested inside gather parameters must become hashable tuples."""
    return tuple(tuple_params(item) for item in value) if isinstance(value, list) else value


def input_records(manifest):
    records = manifest.get("inputs")
    if records:
        return records
    return [{"name": "image", "range": manifest["input_range"], "frame": 0,
             "encoding": manifest["seed_encoding"]}]


def compile_saved_graph(manifest, **compiler_options):
    """Replay saved nodes inside tracing, substituting the supplied input nodes."""
    records = input_records(manifest)
    names = [record["name"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Saved input names must be unique")
    graph = manifest["graph"]
    for index, record in enumerate(graph):
        if record["id"] != index or any(type(arg) is not int or not 0 <= arg < index for arg in record["args"]):
            raise ValueError("Saved graph must have consecutive topologically ordered node IDs")
    if {record["params"][0] for record in graph if record["op"] == "input"} != set(names):
        raise ValueError("Saved graph inputs differ from manifest inputs")
    outputs = manifest["graph_outputs"]
    if not outputs or any(type(index) is not int or not 0 <= index < len(graph) for index in outputs.values()):
        raise ValueError("Saved graph output index out of range")

    def replay(*args, **kwargs):
        supplied = kwargs if kwargs else {names[0]: args[0]}
        replayed = []
        for record in graph:
            if record["op"] == "input":
                image = supplied[record["params"][0]]
            else:
                image = node(record["op"], tuple(replayed[index] for index in record["args"]),
                             tuple_params(record["params"]), *record["range"])
            replayed.append(image)
        return {name: replayed[index] for name, index in outputs.items()}

    options = {"width": manifest["width"], "height": manifest["height"]}
    if manifest.get("input_range") is not None:
        if names != ["image"]:
            raise ValueError("Single-input manifests must use the image input name")
        options["input_range"] = tuple(manifest["input_range"])
    else:
        options["input_ranges"] = {record["name"]: tuple(record["range"]) for record in records}
    options.update(compiler_options)
    return h264.compile(replay, **options)


def extract_inputs(raw, manifest, named):
    images = {}
    for record in input_records(manifest):
        frame = record["frame"]
        if type(frame) is not int or not 0 <= frame < manifest["pictures"]:
            raise ValueError("Saved input picture index out of range")
        board = TOOLS.active_plane(raw, frame, manifest["width"], manifest["height"], manifest["guard"])
        if record["encoding"] == "binary 0/255":
            if any(value not in (0, 255) for row in board for value in row):
                raise AssertionError("Historical binary input contains nonbinary PCM pixels")
            board = [[value // 255 for value in row] for row in board]
        elif record["encoding"] != "literal u8":
            raise ValueError("Unknown saved input encoding: " + record["encoding"])
        images[record["name"]] = board
    return images if named else images["image"]


def annexb_units(stream):
    """Return (prefix start, unit end, NAL kind) for either Annex-B prefix size."""
    starts = list(re.finditer(b"\x00\x00\x00\x01|\x00\x00\x01", stream))
    if not starts or starts[0].start() != 0:
        raise ValueError("Expected an Annex-B stream beginning with a start code")
    units = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(stream)
        if match.end() >= end or stream[match.end()] & 0x80:
            raise ValueError("Invalid Annex-B NAL header")
        units.append((match.start(), end, stream[match.end()] & 31))
    return units


def split_segments(stream):
    """The saved writer emits SPS/PPS and an IDR at each independent job start."""
    units = annexb_units(stream)
    starts = [start for start, _, kind in units if kind == 7]
    if not starts or starts[0] != 0:
        raise ValueError("Each historical stream must start with SPS")
    ends = starts[1:] + [len(stream)]
    result = [stream[start:end] for start, end in zip(starts, ends)]
    for segment in result:
        vcl = [kind for _, _, kind in annexb_units(segment) if kind in (1, 5)]
        if not vcl or vcl[0] != 5:
            raise ValueError("Each historical graph segment must begin with an IDR picture")
    return result


def stream_costs(stream, manifest):
    """Count VCL NALs; committed PCM inputs each occupy exactly one slice."""
    units = annexb_units(stream)
    vcl = [unit for unit in units if unit[2] in (1, 5)]
    inputs = len(input_records(manifest))
    if len(vcl) < inputs:
        raise AssertionError("Stream has fewer slices than input pictures")
    # Parameter sets belong to input loading. Prediction bytes begin after the
    # last single-slice PCM input and include the complete remaining program.
    prediction_start = vcl[inputs - 1][1]
    return {"pictures": manifest["pictures"], "input_pictures": inputs,
            "prediction_slices": len(vcl) - inputs,
            "refresh_copies": len(manifest["refresh_frames"]),
            "stream_bytes": len(stream), "prediction_bytes": len(stream) - prediction_start,
            "peak_live_values": manifest["stats"]["peak_live_values"]}


def sum_costs(cases, key):
    costs = [case["comparison"][key] for case in cases]
    if not costs:
        return {}
    return {name: (max(item[name] for item in costs) if name == "peak_live_values"
                   else sum(item[name] for item in costs)) for name in costs[0]}


def prepare(output, cases=None, local_backends=("ffmpeg", "vaapi", "openh264"), baseline="historical"):
    selected = selected_cases(cases)
    if baseline not in ("historical", "no-exact-search"):
        raise ValueError("Baseline must be historical or no-exact-search")
    if any(backend not in ("ffmpeg", "vaapi", "openh264") for backend in local_backends):
        raise ValueError("Local decoders must be ffmpeg, vaapi or openh264")
    backends = tuple(dict.fromkeys(("ffmpeg", *local_backends)))
    status = {name: {"status": "pending" if available(name) else "unavailable", "cases": 0}
              for name in backends}
    if status["ffmpeg"]["status"] == "unavailable":
        raise RuntimeError("FFmpeg is required for historical and optimized reference pictures")
    bundle = Bundle(Path(output))
    sources = {}
    representatives = {}
    for saved in selected:
        source = REPOSITORY / saved.stream
        manifest_path = REPOSITORY / saved.manifest
        source_bytes, manifest_bytes = source.read_bytes(), manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("guard") != 16:
            raise ValueError("Historical compiler guard must be 16 pixels")
        if manifest.get("stream_sha256") and manifest["stream_sha256"] != sha256(source_bytes):
            raise AssertionError(saved.name + ": historical stream hash differs from saved manifest")
        segments = split_segments(source_bytes)
        if len(segments) != saved.segments:
            raise AssertionError(saved.name + ": historical graph segment count changed")
        raw = decode(source, "ffmpeg")
        frame_bytes = (manifest["width"] + 32) * (manifest["height"] + 32) * 3 // 2
        chunk_bytes = manifest["pictures"] * frame_bytes
        if len(raw) != len(segments) * chunk_bytes:
            raise AssertionError(saved.name + ": historical decoded picture count differs from manifest")
        program = compile_saved_graph(manifest)
        baseline_program = compile_saved_graph(manifest, exact_search=False) if baseline == "no-exact-search" else None
        for index, historical in enumerate(segments):
            name = saved.name if len(segments) == 1 else "{}-{:02d}".format(saved.name, index)
            original_raw = raw[index * chunk_bytes:(index + 1) * chunk_bytes]
            expected = TOOLS.verify_saved_graph(original_raw, manifest)
            inputs = extract_inputs(original_raw, manifest, program.named_inputs)
            if program.reference(inputs) != expected:
                raise AssertionError(name + ": recompiled graph differs from historical graph semantics")
            baseline_bytes, baseline_manifest, baseline_decoded = historical, manifest, original_raw
            baseline_name = name if baseline_program is not None else saved.name
            baseline_stream = bundle.output / "baseline" / (baseline_name + ".h264")
            if baseline_program is not None:
                baseline_program.save(baseline_stream, inputs)
                baseline_bytes = baseline_stream.read_bytes()
                baseline_manifest = json.loads(baseline_stream.with_suffix(".json").read_text())
                baseline_decoded = decode(baseline_stream, "ffmpeg")
                if baseline_program.read(baseline_decoded) != expected:
                    raise AssertionError(name + ": no-exact-search FFmpeg named outputs differ from historical outputs")
            stream = program.save(bundle.output / "streams" / (name + ".h264"), inputs)
            optimized_manifest = json.loads(stream.with_suffix(".json").read_text())
            optimized_raw = decode(stream, "ffmpeg")
            if program.read(optimized_raw) != expected:
                raise AssertionError(name + ": optimized FFmpeg named outputs differ from historical outputs")
            for backend in backends:
                if status[backend]["status"] == "unavailable":
                    continue
                decoded = optimized_raw if backend == "ffmpeg" else decode(stream, backend)
                if decoded != optimized_raw:
                    raise AssertionError(name + ": " + backend + " differs from optimized FFmpeg I420 bytes")
                status[backend]["cases"] += 1
            comparison = {"baseline": baseline, "before": stream_costs(baseline_bytes, baseline_manifest),
                          "after": stream_costs(stream.read_bytes(), optimized_manifest),
                          "baseline_segment_sha256": sha256(baseline_bytes),
                          "baseline_decoded_sha256": sha256(baseline_decoded),
                          "historical_segment_sha256": sha256(historical),
                          "historical_decoded_sha256": sha256(original_raw),
                          "named_outputs_equal": True, "independent_graph_equal": True}
            if baseline_program is not None:
                comparison["baseline_stream"] = str(baseline_stream.relative_to(bundle.output))
                comparison["baseline_manifest"] = str(baseline_stream.with_suffix(".json").relative_to(bundle.output))
            entry = bundle.add(name, "optimized-comparison", stream, optimized_manifest, optimized_raw, expected,
                       group=saved.group, graph=saved.name, segment_index=index,
                       source_stream=saved.stream, source_manifest=saved.manifest,
                       items=saved.items_per_segment, comparison=comparison,
                       oracle_input_source="historical decoded PCM inputs",
                       local_decoders={backend: "pass" if status[backend]["status"] != "unavailable"
                                       else "unavailable" for backend in backends})
            if index == 0:
                baseline_raw = baseline_stream.with_suffix(".i420")
                baseline_stream.parent.mkdir(parents=True, exist_ok=True)
                baseline_stream.write_bytes(baseline_bytes)
                baseline_raw.write_bytes(baseline_decoded)
                representatives[saved.name] = {
                    "group": saved.group, "width": manifest["width"], "height": manifest["height"],
                    "coded_width": manifest["width"] + 32, "coded_height": manifest["height"] + 32,
                    "frame_bytes": frame_bytes,
                    "baseline": {"stream": str(baseline_stream.relative_to(bundle.output)),
                                 "expected": str(baseline_raw.relative_to(bundle.output)),
                                 "stream_sha256": sha256(baseline_bytes), "expected_sha256": sha256(baseline_decoded),
                                 "pictures": baseline_manifest["pictures"], "expected_bytes": len(baseline_decoded)},
                    "optimized": {key: entry[key] for key in ("name", "stream", "expected", "stream_sha256",
                                                               "expected_sha256", "pictures", "expected_bytes")}}
        if source.read_bytes() != source_bytes or manifest_path.read_bytes() != manifest_bytes:
            raise AssertionError(saved.name + ": historical source changed during comparison")
        sources[saved.stream] = sha256(source_bytes)
        sources[saved.manifest] = sha256(manifest_bytes)
    for record in status.values():
        if record["status"] != "unavailable":
            record["status"] = "pass"
    groups = {}
    for group in dict.fromkeys(case["group"] for case in bundle.cases):
        entries = [case for case in bundle.cases if case["group"] == group]
        groups[group] = {"cases": len(entries), "items": sum(case["items"] for case in entries),
                         "before": sum_costs(entries, "before"), "after": sum_costs(entries, "after")}
    summary = {"baseline": baseline, "cases": len(bundle.cases), "graphs_compiled": len(selected), "groups": groups,
               "before": sum_costs(bundle.cases, "before"), "after": sum_costs(bundle.cases, "after"),
               "named_outputs": sum(len(case["named_outputs"]) for case in bundle.cases),
               "named_output_pixels": sum(case["width"] * case["height"] * len(case["named_outputs"])
                                           for case in bundle.cases), "local_backends": status,
               "representatives": representatives}
    version = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    source_paths = [*sorted((COMPILER / "h264").glob("*.py")), REPOSITORY / "decoder_core" / "bitstream.py",
                    REPOSITORY / "decoder_core" / "decode.py", Path(__file__).resolve()]
    suite = {"format": "h264-nvdec-suite-0.1", "created_utc": datetime.now(timezone.utc).isoformat(),
             "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                                                           text=True).strip(),
             "python": platform.python_version(), "reference_decoder": version,
             "raw_format": "I420", "raw_order": "display order; full coded pictures including guard and chroma",
             "oracle": "historical FFmpeg named outputs and independent saved-graph integer evaluation",
             "baseline": baseline,
             "comparison": ("Current compiler exact_search=False vs exact_search=True; all other optimizations unchanged; "
                            "compile each variant once per saved graph, encode every original segment"
                            if baseline == "no-exact-search" else
                            "Historical streams vs default optimized compiler; compile once per saved graph, encode every original segment"),
             "cost_definitions": {"prediction_slices": "NAL kind 1 or 5 count minus one PCM slice per input",
                                  "prediction_bytes": "Bytes after the final PCM input NAL, including the full prediction tail",
                                  "peak_live_values": "Per-graph scheduled live values; aggregate uses maximum",
                                  "mnist_items": "16 digits in two independent eight-digit packed graph executions"},
             "historical_source_sha256": sources,
             "compiler_source_sha256": {str(path.relative_to(REPOSITORY)): sha256(path.read_bytes())
                                         for path in source_paths},
             "case_count": len(bundle.cases), "total_pictures": sum(case["pictures"] for case in bundle.cases),
             "total_expected_bytes": sum(case["expected_bytes"] for case in bundle.cases),
             "summary": summary, "representatives": representatives, "cases": bundle.cases}
    write_json(bundle.output / "suite.json", suite)
    write_json(bundle.output / "comparison.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "bundle")
    parser.add_argument("--cases", nargs="+", help="Groups life maze denoise mnist native motion, or native-NAME")
    parser.add_argument("--local-backends", nargs="+", choices=("ffmpeg", "vaapi", "openh264"),
                        default=["ffmpeg", "vaapi", "openh264"])
    parser.add_argument("--baseline", choices=("historical", "no-exact-search"), default="historical",
                        help="Compare historical streams or current compiler with exact instruction search disabled")
    args = parser.parse_args()
    try:
        selected_cases(args.cases)
    except ValueError as error:
        parser.error(str(error))
    prepare(args.output, args.cases, args.local_backends, args.baseline)


if __name__ == "__main__":
    main()
