#!/usr/bin/env python3
"""Compile a reproducible motion video and verify it before NVDEC transfer."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from fixtures import sequence
from kernel import compile_motion
from reference import motion
from decoder_core.decode import available, decode


ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plane(board):
    return bytes(value for row in board for value in row)


def pgm(path, board):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("P5\n%d %d\n255\n" % (len(board[0]), len(board))).encode()+plane(board))


def prepare(output, jobs=24, decoders=("ffmpeg", "vaapi", "openh264")):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    program = compile_motion()
    compile_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    tail = program.tail
    tail_ms = (time.perf_counter()-started)*1000
    frames = sequence(jobs)
    frame_bytes = program.writer.width*program.writer.height*3//2
    report = {"format": "h264-motion-bundle-0.1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "source_sha256": {str(path.relative_to(ROOT.parent)): digest(path.read_bytes())
                                for path in sorted([*ROOT.glob("*.py"),
                                    *(ROOT.parent/"05-h264-compiler"/"h264").glob("*.py"),
                                    ROOT.parent/"decoder_core"/"bitstream.py",
                                    ROOT.parent/"decoder_core"/"decode.py"])},
              "width": program.width, "height": program.height, "coded_width": program.writer.width,
              "coded_height": program.writer.height, "frame_bytes": frame_bytes, "guard": 16,
              "cutoff": 32, "pictures_per_job": program.pictures, "output_frames": program.output_frames,
              "prediction_sha256": digest(tail), "prediction_bytes_per_job": len(tail),
              "compile_ms": compile_ms, "first_tail_ms": tail_ms, "jobs": [],
              "decoders": {}, "input": "deterministic synthetic video; moving shapes and sensor-like noise"}
    (output/"program.json").write_text(json.dumps(program.manifest(), indent=2)+"\n")
    for index, board in enumerate(frames):
        pgm(output/"inputs"/("%03d.pgm" % index), board)
    selected_backends = []
    for backend in decoders:
        ready = available(backend)
        report["decoders"][backend] = {"status": "pending" if ready else "unavailable", "jobs": 0}
        if ready:
            selected_backends.append(backend)
    if "ffmpeg" not in selected_backends:
        raise RuntimeError("FFmpeg is required to prepare full-picture reference bytes")
    for index in range(jobs):
        inputs = {"previous": frames[index], "current": frames[index+1]}
        expected = motion(**inputs)
        if program.reference(inputs) != expected:
            raise AssertionError("Compiler graph differs from independent motion oracle")
        started = time.perf_counter()
        seed = program.encode_inputs(inputs)
        packing_ms = (time.perf_counter()-started)*1000
        stream = output/"streams"/("%03d.h264" % index)
        stream.parent.mkdir(parents=True, exist_ok=True)
        stream.write_bytes(seed+tail)
        raw = decode(stream, "ffmpeg")
        if program.read(raw) != expected:
            raise AssertionError("FFmpeg differs from independent motion oracle")
        for backend in selected_backends:
            decoded = raw if backend == "ffmpeg" else decode(stream, backend)
            if decoded != raw or program.read(decoded) != expected:
                raise AssertionError("%s differs at job %d" % (backend, index))
            report["decoders"][backend]["jobs"] += 1
        expected_path = output/"expected"/("%03d.i420" % index)
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(raw)
        named = {}
        for label, board in expected.items():
            named[label] = {"frame": program.output_frames[label], "sha256": digest(plane(board)),
                            "nonzero_pixels": sum(value != 0 for row in board for value in row)}
        report["jobs"].append({"index": index, "stream": str(stream.relative_to(output)),
                                "stream_sha256": digest(stream.read_bytes()), "stream_bytes": stream.stat().st_size,
                                "input_bytes": len(seed), "input_packing_ms": packing_ms,
                                "expected": str(expected_path.relative_to(output)), "expected_sha256": digest(raw),
                                "named_outputs": named})
        print("PASS job %02d: %d pictures; all named outputs + full raw bytes" % (index, program.pictures), flush=True)
    for backend in selected_backends:
        report["decoders"][backend]["status"] = "pass"
    report["job_count"] = jobs
    report["total_pictures"] = jobs*program.pictures
    report["total_raw_bytes"] = jobs*program.pictures*frame_bytes
    complete = b"".join((output/job["stream"]).read_bytes() for job in report["jobs"])
    (output/"motion.h264").write_bytes(complete)
    report["combined_stream_sha256"] = digest(complete)
    report["combined_stream_bytes"] = len(complete)
    (output/"bundle.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"out"/"bundle")
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--decoders", nargs="+", choices=("ffmpeg", "vaapi", "openh264"),
                        default=["ffmpeg", "vaapi", "openh264"])
    args = parser.parse_args()
    result = prepare(args.output, args.jobs, args.decoders)
    print(json.dumps({key: result[key] for key in ("job_count", "total_pictures", "total_raw_bytes", "decoders")}))


if __name__ == "__main__":
    main()
