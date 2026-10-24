#!/usr/bin/env python3
"""Build a portable NVDEC decoder suite, verified locally before transfer.

Only this preparation step imports the repository compiler and demo fixtures.
The resulting bundle contains ordinary files usable by a stdlib-only runner:
Annex B streams, manifests, full FFmpeg I420 reference bytes, and independently
evaluated named luma outputs. No compiler or Pillow installation is needed on
the target. Run from any directory with ``--output PATH``.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace


COMPILER = Path(__file__).resolve().parents[2]
NATIVE_OPS = COMPILER / "native_ops"
REPOSITORY = COMPILER.parent
sys.path[:0] = [str(COMPILER), str(NATIVE_OPS)]

import h264
from benchmark import inputs as benchmark_inputs, kernels as benchmark_kernels
from decoder_core.decode import decode
from demos import cases as demo_cases
from h264.graph import Image
from h264.verify import evaluate, read


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def plane_bytes(board):
    return bytes(value for row in board for value in row)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def random_board(width, height, seed):
    rng = random.Random(seed)
    return [[rng.randrange(256) for _ in range(width)] for _ in range(height)]


def active_plane(raw, frame, width, height, guard=16):
    coded_width, coded_height = width + 2*guard, height + 2*guard
    start = frame*coded_width*coded_height*3//2
    return [list(raw[start+(y+guard)*coded_width+guard:
                     start+(y+guard)*coded_width+guard+width]) for y in range(height)]


def verify_saved_graph(raw, manifest):
    """Re-evaluate committed graph semantics using its decoded PCM input.

    This handles older saved compiler manifests without importing their demo
    packages or regenerating model assets. Decoder output supplies only the
    input picture; all named output arithmetic uses the integer graph oracle.
    """
    graph = []
    for record in manifest["graph"]:
        graph.append(Image(record["op"], tuple(graph[index] for index in record["args"]),
                           tuple(record["params"]), *record["range"]))
    width, height, guard = manifest["width"], manifest["height"], manifest["guard"]
    if guard != 16:
        raise ValueError("Committed compiler guard must be 16 pixels")
    proxy = SimpleNamespace(width=width, height=height, pictures=manifest["pictures"],
                            writer=SimpleNamespace(width=width+2*guard, height=height+2*guard),
                            graph=graph, graph_outputs=manifest["graph_outputs"],
                            output_frames=manifest["output_frames"])
    actual = read(raw, proxy)
    input_records = {record["name"]: record for record in manifest.get("inputs", [])}
    inputs = {}
    for item in graph:
        if item.op != "input":
            continue
        name = item.params[0]
        record = input_records.get(name, {"frame": 0, "encoding": manifest["seed_encoding"]})
        board = active_plane(raw, record["frame"], width, height, guard)
        if record["encoding"] == "binary 0/255":
            if any(value not in (0, 255) for row in board for value in row):
                raise AssertionError("Committed binary PCM input contains nonbinary pixels")
            board = [[value//255 for value in row] for row in board]
        inputs[name] = board
    expected = evaluate(graph, manifest["graph_outputs"], inputs)
    if actual != expected:
        raise AssertionError("Committed stream differs from its independent graph semantics")
    return expected


class Bundle:
    def __init__(self, output):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.cases = []

    def add(self, name, kind, stream, manifest, raw, expected, **details):
        width, height = manifest["width"], manifest["height"]
        guard = manifest.get("guard", 16)
        coded_width, coded_height = width+2*guard, height+2*guard
        frame_bytes = coded_width*coded_height*3//2
        if len(raw) != manifest["pictures"]*frame_bytes:
            raise AssertionError(f"{name}: full I420 picture count differs from manifest")
        raw_path = self.output / "expected" / f"{name}.i420"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        output_hashes, named_outputs = {}, {}
        for index, (label, board) in enumerate(expected.items()):
            values = plane_bytes(board)
            if len(values) != width*height:
                raise AssertionError(f"{name}/{label}: independent output dimensions differ")
            frame = manifest["output_frames"][label]
            if plane_bytes(active_plane(raw, frame, width, height, guard)) != values:
                raise AssertionError(f"{name}/{label}: FFmpeg differs from independent pixels")
            path = self.output / "named_outputs" / name / f"{index:03d}.u8"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(values)
            output_hashes[label] = sha256(values)
            named_outputs[label] = {"path": str(path.relative_to(self.output)),
                                    "bytes": len(values), "sha256": sha256(values), "frame": frame}
        entry = {"name": name, "kind": kind, "width": width, "height": height,
                 "coded_width": coded_width, "coded_height": coded_height, "guard": guard,
                 "pictures": manifest["pictures"], "frame_bytes": frame_bytes,
                 "stream": str(stream.relative_to(self.output)),
                 "manifest": str(stream.with_suffix(".json").relative_to(self.output)),
                 "expected": str(raw_path.relative_to(self.output)),
                 "expected_bytes": len(raw), "expected_sha256": sha256(raw),
                 "stream_bytes": stream.stat().st_size, "stream_sha256": sha256(stream.read_bytes()),
                 "output_frames": manifest["output_frames"], "named_output_sha256": output_hashes,
                 "named_outputs": named_outputs, "local_ffmpeg_oracle": "pass", **details}
        self.cases.append(entry)
        print(f"PASS {name}: {entry['pictures']} pictures; {len(expected)} named outputs", flush=True)
        return entry

    def compiled(self, name, kind, program, inputs, **details):
        stream = program.save(self.output / "streams" / f"{name}.h264", inputs)
        expected = program.reference(inputs)
        raw = decode(stream, "ffmpeg")
        if program.read(raw) != expected:
            raise AssertionError(f"{name}: FFmpeg differs from independent graph semantics")
        return self.add(name, kind, stream, json.loads(stream.with_suffix(".json").read_text()),
                        raw, expected, refresh_copies=len(program.refreshes), **details)

    def committed(self, name, relative_stream, relative_manifest):
        source = REPOSITORY / relative_stream
        manifest = json.loads((REPOSITORY / relative_manifest).read_text())
        stream = self.output / "streams" / f"{name}.h264"
        stream.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, stream)
        if manifest.get("stream_sha256") and manifest["stream_sha256"] != sha256(stream.read_bytes()):
            raise AssertionError(f"{name}: committed manifest stream hash differs")
        raw = decode(stream, "ffmpeg")
        chunk_bytes = manifest["pictures"]*(manifest["width"]+32)*(manifest["height"]+32)*3//2
        if len(raw) % chunk_bytes:
            raise AssertionError(f"{name}: committed stream contains an incomplete graph execution")
        segments = len(raw)//chunk_bytes
        expected, output_frames = {}, {}
        for index in range(segments):
            outputs = verify_saved_graph(raw[index*chunk_bytes:(index+1)*chunk_bytes], manifest)
            for label, board in outputs.items():
                key = label if segments == 1 else f"segment_{index:02d}/{label}"
                expected[key] = board
                output_frames[key] = index*manifest["pictures"]+manifest["output_frames"][label]
        if segments != 1:
            manifest = {"format": "h264-concatenated-graphs-0.1", "width": manifest["width"],
                        "height": manifest["height"], "guard": manifest["guard"],
                        "pictures": segments*manifest["pictures"], "output_frames": output_frames,
                        "segments": segments, "segment_pictures": manifest["pictures"],
                        "segment_manifest": manifest, "stream_sha256": sha256(stream.read_bytes())}
        write_json(stream.with_suffix(".json"), manifest)
        return self.add(name, "legacy-compiled", stream, manifest, raw, expected,
                        source_stream=relative_stream, source_manifest=relative_manifest,
                        oracle_input_source="decoded PCM seed", graph_executions=segments)


def prepare_correctness(bundle):
    for name, title, kernel, options, inputs in demo_cases():
        bundle.compiled(f"demo-{name}", "native-demo", h264.compile(kernel, **options), inputs, title=title)

    named = {"width": 64, "height": 64, "input_ranges": {"a": (0, 255), "b": (0, 255)}}
    addition = h264.compile(h264.saturating_add, **named)
    for chunk in range(16):
        a = [[(chunk*4096+y*64+x)//256 for x in range(64)] for y in range(64)]
        b = [[(y*64+x) % 256 for x in range(64)] for y in range(64)]
        bundle.compiled(f"exhaustive-add-{chunk:02d}", "exhaustive-u8-add", addition, {"a": a, "b": b},
                        pair_start=chunk*4096, pair_count=4096)

    offsets = [(fx, fy) for fy in range(4) for fx in range(4)]
    offsets += [(fx-4, fy-4) for fy in range(4) for fx in range(4)]
    offsets += [(67, -67), (-67, 67), (508, 508), (-508, -508)]
    def phases(image):
        return {f"q{qx}_{qy}": h264.sample(image, qx=qx, qy=qy) for qx, qy in offsets}
    bundle.compiled("all-quarter-phases", "interpolation", h264.compile(phases, 32, 32, input_range=(0, 255)),
                    random_board(32, 32, 241), quarter_offsets=offsets)

    gather_offsets = [(fx-4, fy-4) for fy in range(4) for fx in range(4)]
    gather = h264.compile(lambda image: h264.gather(image, offsets=gather_offsets), 64, 64, input_range=(0, 255))
    bundle.compiled("fractional-block-gather", "gather", gather, random_board(64, 64, 242),
                    block_offsets=gather_offsets)

    weights = [(1, -1, 127), (63, -64, 127), (-64, 63, -128), (-1, -1, 127), (2, 3, -127)]
    def signed(a, b):
        a = h264.sample(a, qx=1, qy=2)
        b = h264.sample(b, qx=-3, qy=-1)
        return {f"weights_{wa}_{wb}_{bias}": h264.saturating_combine(a, b, wa, wb, bias)
                for wa, wb, bias in weights}
    named32 = {"width": 32, "height": 32, "input_ranges": {"a": (0, 255), "b": (0, 255)}}
    images = {"a": random_board(32, 32, 243), "b": random_board(32, 32, 244)}
    bundle.compiled("signed-b-fractional", "signed-b-weighted", h264.compile(signed, **named32), images,
                    weights=weights)

    def refresh(a, b):
        value = a
        for index in range(75):
            value = h264.saturating_combine(value, a if index % 2 else b, wb=-1, bias=67)
        return {"output": value, "original_a": a, "original_b": b}
    refreshed = h264.compile(refresh, **named32)
    if not refreshed.refreshes:
        raise AssertionError("Long-chain fixture did not exercise reference refresh")
    bundle.compiled("resident-75-stages", "dpb-refresh", refreshed, images, logical_operations=75)

    binary = [[(x*7+y*11) % 5 == 0 for x in range(32)] for y in range(32)]
    binary = [[int(value) for value in row] for row in binary]
    shifted_or = h264.compile(lambda image: h264.bit_or(h264.shift(image, 1, 0), h264.shift(image, 0, 1)), 32, 32)
    bundle.compiled("binary-shifted-or", "binary-compatibility", shifted_or, binary)

    committed = [
        ("compiled-life", "01-conways-decoder-of-life/compiled/assets/life.h264", "life.json"),
        ("compiled-maze", "02-maze-flood-fill-decoder-solver/compiled/assets/maze.h264", "maze.json"),
        ("compiled-denoiser", "03-denoising-decoder/compiled/assets/denoising.h264", "program.json"),
        ("compiled-mnist", "04-binary-mnist-classification-decoder/compiled/assets/mnist.h264", "program.json"),
        ("committed-shifted-or", "05-h264-compiler/assets/shifted-or.h264", "shifted-or.json"),
    ]
    for name, stream, manifest_name in committed:
        bundle.committed(name, stream, str(Path(stream).with_name(manifest_name)))


def prepare_benchmarks(bundle):
    config = json.loads((NATIVE_OPS / "benchmark.json").read_text())["configuration"]
    width, height, steps = config["width"], config["height"], config["steps"]
    if steps != 32:
        raise AssertionError("Committed benchmark must contain 32 logical additions")
    pair_kernel, resident_kernel = benchmark_kernels(steps)
    options = {"width": width, "height": height, "input_ranges": {"a": (0, 255), "b": (0, 255)}}
    pair, resident = h264.compile(pair_kernel, **options), h264.compile(resident_kernel, **options)
    inputs = benchmark_inputs(width, height)
    for name, program in (("benchmark-resident", resident), ("benchmark-fresh-one-session", pair)):
        source = NATIVE_OPS / "streams" / f"{name}.h264"
        stream = bundle.output / "streams" / source.name
        shutil.copyfile(source, stream)
        raw = decode(stream, "ffmpeg")
        if name == "benchmark-resident":
            manifest = json.loads(source.with_suffix(".json").read_text())
            expected = resident.reference(inputs)
            if resident.read(raw) != expected:
                raise AssertionError("Committed resident benchmark differs from integer reference")
        else:
            frame_bytes = pair.writer.width*pair.writer.height*3//2
            if len(raw) % frame_bytes:
                raise AssertionError("Concatenated benchmark contains a partial I420 frame")
            pictures = len(raw)//frame_bytes
            if pictures != steps*pair.pictures or pictures != 96:
                raise AssertionError("Fresh one-session benchmark must contain 96 pictures")
            expected, output_frames = {}, {}
            current = inputs["a"]
            chunk_bytes = pair.pictures*frame_bytes
            for index in range(steps):
                payload = {"a": current, "b": inputs["b"]}
                reference = pair.reference(payload)
                if pair.read(raw[index*chunk_bytes:(index+1)*chunk_bytes]) != reference:
                    raise AssertionError(f"Fresh benchmark addition {index+1} differs from integer reference")
                current = reference["output"]
                label = f"step_{index+1:02d}"
                expected[label] = current
                output_frames[label] = index*pair.pictures+pair.output_frames["output"]
            expected["output"] = current
            output_frames["output"] = pictures-1
            manifest = {"format": "h264-concatenated-benchmark-0.1", "width": width, "height": height,
                        "guard": 16, "pictures": pictures, "output_frames": output_frames,
                        "segments": steps, "segment_pictures": pair.pictures,
                        "segment_manifest": pair.manifest(), "stream_sha256": sha256(stream.read_bytes())}
        write_json(stream.with_suffix(".json"), manifest)
        bundle.add(name, "benchmark", stream, manifest, raw, expected, logical_operations=steps,
                   decoder_sessions=1, source_stream=str(source.relative_to(REPOSITORY)),
                   refresh_copies=len(program.refreshes),
                   workload="resident" if name == "benchmark-resident" else "fresh-one-session")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Directory containing the portable suite")
    args = parser.parse_args()
    bundle = Bundle(args.output)
    prepare_correctness(bundle)
    prepare_benchmarks(bundle)
    version = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
                            capture_output=True, text=True, check=True).stdout.strip()
    suite = {"format": "h264-nvdec-suite-0.1", "created_utc": datetime.now(timezone.utc).isoformat(),
             "repository_commit": commit, "python": platform.python_version(), "reference_decoder": version,
             "raw_format": "I420", "raw_order": "display order; full coded pictures including guard and chroma",
             "oracle": "independent integer graph evaluation; full raw bytes supplied by FFmpeg",
             "case_count": len(bundle.cases), "total_pictures": sum(case["pictures"] for case in bundle.cases),
             "total_expected_bytes": sum(case["expected_bytes"] for case in bundle.cases),
             "cases": bundle.cases}
    write_json(bundle.output / "suite.json", suite)
    print(f"Prepared {suite['case_count']} cases, {suite['total_pictures']} pictures, "
          f"{suite['total_expected_bytes']:,} expected bytes: {bundle.output}")


if __name__ == "__main__":
    main()
