"""Prepare independently checked H.264 temporal workloads for portable timing.

This exporter compiles once, saves distinct deterministic input pairs, and checks
all named outputs against independent task formulas and the semantic IR before
freezing full FFmpeg I420 references. Timing lives in temporal_benchmark.py.
"""

import argparse
import json
from pathlib import Path
import subprocess

import codec_compute as media
from codec_compute.targets.h264 import _legacy
from decoder_core.decode import decode_batch
from .kernels import WORKLOADS, reference
from .temporal_benchmark import TASKS, digest, geometry, oracle, validate_guards, write_json


ROOT = Path(__file__).resolve().parents[2]


def input_pair(index, width, height):
    """Distinct deterministic byte pairs, including equality, limits and motion.

    A local integer PRNG avoids version-sensitive random/library behavior. These
    are independent standalone jobs: no decoded surface is reused across jobs.
    """
    state = 0xC0DEC0DE ^ (index*0x9E3779B9)
    planes = []
    for image in range(2):
        values = bytearray()
        for y in range(height):
            for x in range(width):
                state = (1664525*state+1013904223) & 0xffffffff
                value = (state >> 24) if index % 2 else (x*3+y*5+index*17) % 128
                if index % 2 == 0 and width//4+image*3 <= x < width//2+image*3 and height//4 <= y < 3*height//4:
                    value = 210
                values.append(value)
        planes.append(values)
    boundaries = ((0, 0), (255, 255), (255, 0), (0, 255), (23, 0), (24, 0), (0, 24), (1, 0))
    for i, (a, b) in enumerate(boundaries):
        planes[0][i], planes[1][i] = a, b
    return {name: [list(values[y*width:(y+1)*width]) for y in range(height)]
            for name, values in zip(("a", "b"), planes)}


def prepare(output, *, width=128, height=128, jobs=16):
    if type(jobs) is not int or jobs < 2:
        raise ValueError("Prepare at least two independent jobs")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = sorted(path for path in (ROOT / "codec_compute").rglob("*.py")
                     if not {"tests", "artifacts", "assets"}.intersection(path.parts))
    sources += sorted((ROOT / "05-h264-compiler" / "h264").glob("*.py"))
    sources += [ROOT / "decoder_core" / name for name in ("bitstream.py", "decode.py")]
    source_hashes = {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}
    inputs = [input_pair(index, width, height) for index in range(jobs)]

    def artifact(relative, data):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": relative, "sha256": digest(data), "bytes": len(data)}

    def plane(relative, board):
        return artifact(relative, bytes(v for row in board for v in row))

    input_records = [{name: plane("inputs/%03d-%s.u8" % (index, name), board)
                      for name, board in images.items()} for index, images in enumerate(inputs)]
    tasks = {}
    for task in (task for task in WORKLOADS if task.name in TASKS):
        program = media.compile(task.kernel, target="h264", width=width, height=height, input_ranges=task.input_ranges)
        manifest, tail = program.manifest(), program.tail
        shape = geometry(manifest)
        manifest_record = artifact(task.name + "/manifest.json", (json.dumps(manifest, indent=2)+"\n").encode())
        tail_record = artifact(task.name + "/prediction.h264", tail)
        paths, records, expected = [], [], []
        for index, images in enumerate(inputs):
            outputs = reference(task, images)
            if program.reference(images) != outputs:
                raise AssertionError("Semantic IR differs from independent task formula")
            native_bytes = {name: bytes(v for row in board for v in row) for name, board in images.items()}
            if oracle(task.name, native_bytes["a"], native_bytes["b"]) != {
                    name: bytes(v for row in board for v in row) for name, board in outputs.items()}:
                raise AssertionError("Portable oracle differs from task formula")
            stream = program.encode(images)
            if not stream.endswith(tail) or program.tail != tail:
                raise AssertionError("Prediction tail depends on inputs")
            stem = "%s/%03d" % (task.name, index)
            records.append({"index": index, "inputs": input_records[index],
                            "input_prefix": artifact(stem + ".prefix", stream[:-len(tail)]),
                            "stream": artifact(stem + ".h264", stream)})
            paths.append(root / records[-1]["stream"]["path"])
            expected.append(outputs)
        decoded = decode_batch(paths, "ffmpeg", width=shape["coded_width"], height=shape["coded_height"],
                               pictures_per_job=shape["pictures_per_job"])
        for index, (record, frames, want) in enumerate(zip(records, decoded.frames, expected)):
            raw = b"".join(frames)
            actual = _legacy().backend.read(raw, program.native)
            if actual != want:
                raise AssertionError("FFmpeg named outputs differ from independent formula")
            validate_guards(frames, shape)
            record["expected"] = artifact("%s/%03d.i420" % (task.name, index), raw)
            record["outputs"] = {name: plane("%s/%03d-%s.u8" % (task.name, index, name), board)
                                 for name, board in actual.items()}
        writer = program.native.writer
        params = {"width": width, "height": height, "fps": writer.fps, "motion_bits": writer.motion_bits,
                  "reference_slots": dict(writer.reference_slots), "slice_mode": writer.slice_mode}
        tasks[task.name] = {"description": task.description, "costs": manifest["costs"],
                            "manifest": manifest_record, "tail": tail_record, "writer": params, "jobs": records}
        print("PASS %s: %d independent jobs, %d pictures/job, %d-byte fixed prediction tail" %
              (task.name, jobs, shape["pictures_per_job"], len(tail)), flush=True)
    if source_hashes != {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in sources}:
        raise AssertionError("Source changed while preparing benchmark")
    runtime = {"runner": "codec_compute/workloads/temporal_benchmark.py",
               "decoder": "decoder_core/decode.py", "writer": "decoder_core/bitstream.py"}
    bundle = {"format": "codec-compute-temporal-bundle-1", "status": "pass", "target": "h264",
              "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": source_hashes,
              "runtime_source_sha256": {name: source_hashes[path] for name, path in runtime.items()},
              "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
              "width": width, "height": height, "job_count": jobs,
              "input_generation": "Independent deterministic byte pairs; local 32-bit LCG plus scene motion and exact arithmetic boundary pixels",
              "validation": "Every named output equals independent task formula and semantic IR; full saved FFmpeg I420 includes all inputs/intermediates, zero guards and neutral chroma",
              "tasks": tasks}
    write_json(root / "bundle.json", bundle)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "codec_compute/artifacts/temporal-benchmark")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()
    prepare(args.output, width=args.width, height=args.height, jobs=args.jobs)
