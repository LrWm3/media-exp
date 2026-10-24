"""Export exact, framed input/output fixtures for a compiler-free runtime test."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import codec_compute as media
from decoder_core.decode import available, decode
from ..resampling.kernels import fixture, reference
from .tasks import tasks
from .native import CPU


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def prepare(output, *, cpu_helper, sizes=(128, 256), jobs=8):
    if not sizes or any(type(size) is not int or size not in (128, 256) for size in sizes):
        raise ValueError("Supported experiment sizes are 128 and 256")
    if type(jobs) is not int or jobs < 2:
        raise ValueError("At least two changing input fixtures are required")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    sources = {"codec_compute/compiler.py", "codec_compute/runtime.py", "codec_compute/targets/h264.py",
               "decoder_core/bitstream.py", "decoder_core/fence.py", "decoder_core/session.py",
               "decoder_core/nvdec_decode.cpp", "decoder_core/decode.py",
               "codec_compute/workloads/resident/prepare.py", "codec_compute/workloads/resident/benchmark.py",
               "codec_compute/workloads/resampling/kernels.py", "codec_compute/workloads/temporal_benchmark.py",
               "codec_compute/workloads/resident/capture.py"}
    for folder in ("codec_compute", "codec_compute/synthesis", "codec_compute/targets", "05-h264-compiler/h264"):
        sources.update(str(path.relative_to(repository)) for path in (repository/folder).rglob("*.py")
                       if "tests" not in path.parts and "artifacts" not in path.parts)
    sources.update(str(path.relative_to(repository)) for path in (repository/"codec_compute/workloads/cpu_comparison").glob("*.py"))
    sources.update({"codec_compute/workloads/cpu_comparison/cpu.cpp", "decoder_core/pcm.py", "decoder_core/pcm_native.cpp", "decoder_core/build_pcm.sh",
                    "codec_compute/workloads/cpu_comparison/build_cpu.sh"})
    sources = sorted(sources)
    report = {"format": "codec-compute-resident-1", "status": "running", "sizes": list(sizes),
              "jobs": jobs, "experiment": "cpu-comparison",
              "cpu_helper_sha256": digest(Path(cpu_helper).read_bytes()), "source_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
              "source_sha256": {name: digest((repository/name).read_bytes()) for name in sources},
              "cases": {}, "validation": {"streams": 0, "pictures": 0, "local_decoder_runs": 0}}
    backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
    if "ffmpeg" not in backends:
        raise RuntimeError("FFmpeg is required")
    report["decoders"] = backends

    def artifact(path, data):
        path = root/path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path.relative_to(root)), "bytes": len(data), "sha256": digest(data)}

    for size in sizes:
        level = 31 if size == 128 else 41
        for task, cpu_spec in tasks(size):
            name, fuse, variant = task.name, True, "fused"
            key = f"{size}/{name}/{variant}"
            program = media.compile(task.kernel, width=size, height=size,
                                    input_ranges=task.input_ranges, h264_level=level,
                                    h264_fuse_shifts=fuse)
            manifest = program.manifest()
            native, writer = program.native, program.native.writer
            coded_w, coded_h = writer.width, writer.height
            frame_bytes = coded_w*coded_h*3//2
            case = {"cpu": cpu_spec, "size": size, "level_idc": level, "input_names": list(task.input_names),
                    "costs": manifest["costs"], "coded_width": coded_w, "coded_height": coded_h,
                    "frame_bytes": frame_bytes, "output_frames": native.output_frames,
                    "writer": {"width": size, "height": size, "fps": writer.fps,
                               "motion_bits": writer.motion_bits, "slice_mode": writer.slice_mode,
                               "level_idc": level, "reference_slots": dict(writer.reference_slots)},
                    "manifest": artifact(Path(key)/"manifest.json", (json.dumps(manifest, indent=2)+"\n").encode()),
                    "prediction_units": [artifact(Path(key)/f"prediction-{i:03}.h264", unit)
                                         for i, unit in enumerate(native.prediction_units)],
                    "fence": artifact(Path(key)/"fence.h264", native.fence_unit), "jobs": []}
            report["cases"][key] = case
            cpu = CPU(cpu_helper, size, cpu_spec)
            for index in range(jobs):
                images = fixture(task, index, size)
                # Exercise escape-heavy and extreme inputs as well as textured
                # scenes/noise, identically across paired interpolation/control.
                if index % 8 in (2, 3):
                    images = {n: [[255 if (index % 8 == 3) ^ bool(i) else 0]*size for _ in range(size)]
                              for i, n in enumerate(task.input_names)}
                compact = {n: bytes(v for row in image for v in row) for n, image in images.items()}
                cpu_output = cpu.run(*(compact[n] for n in task.input_names))
                expected = {"output": [list(cpu_output[y*size:(y+1)*size]) for y in range(size)]}
                # The independent C++ recurrence checks every fixture. Also
                # compare direct scalar task formulas for all shallow scenes.
                if cpu_spec["kind"] != 4:
                    if expected != reference(task, images):
                        raise AssertionError("Native CPU and independent task formula differ")
                units = native.access_units(images)
                if b"".join(units) != program.encode(images):
                    raise AssertionError("AU framing changes the compiled stream")
                stream = b"".join(units) + native.fence_unit
                folder = Path(key)/f"{index:03}"
                stream_record = artifact(folder/"program.h264", stream)
                actual = None
                for backend in backends:
                    raw = decode(root/stream_record["path"], backend)
                    if len(raw) != (native.pictures+1)*frame_bytes:
                        raise AssertionError("Decoded picture count differs")
                    if actual is not None and raw != actual:
                        raise AssertionError("Complete decoder I420 differs")
                    actual = raw
                    report["validation"]["local_decoder_runs"] += 1
                frames = tuple(actual[i:i+frame_bytes] for i in range(0, len(actual), frame_bytes))
                if frames[-1] != bytes([128])*frame_bytes:
                    raise AssertionError("Discarded IDR fence must decode to constant128")
                for frame in frames[:-1]:
                    for y in range(coded_h):
                        row = frame[y*coded_w:(y+1)*coded_w]
                        if any(row if y < 16 or y >= size+16 else row[:16]+row[-16:]):
                            raise AssertionError("Nonzero computation guard")
                    if frame[coded_w*coded_h:] != bytes([128])*(frame_bytes-coded_w*coded_h):
                        raise AssertionError("Non-neutral computation chroma")
                def plane(frame):
                    return bytes(value for y in range(size)
                                 for value in frame[(y+16)*coded_w+16:(y+16)*coded_w+16+size])
                for i, name_in in enumerate(task.input_names):
                    if plane(frames[i]) != bytes(v for row in images[name_in] for v in row):
                        raise AssertionError("Decoded input differs")
                outputs = {}
                for output_name, frame_index in native.output_frames.items():
                    values = plane(frames[frame_index])
                    if values != bytes(v for row in expected[output_name] for v in row):
                        raise AssertionError("Decoded result differs from independent formula")
                    outputs[output_name] = artifact(folder/f"{output_name}.u8", values)
                case["jobs"].append({"inputs": {n: artifact(folder/f"input-{n}.u8",
                                                              bytes(v for row in image for v in row))
                                                    for n, image in images.items()},
                                     "units": [artifact(folder/f"input-{i:03}.h264", unit)
                                               for i, unit in enumerate(units[:len(task.input_names)])],
                                     "stream": stream_record, "ordinary": artifact(folder/"ordinary.h264", b"".join(units)), "expected": artifact(folder/"expected.i420", actual),
                                     "outputs": outputs})
                report["validation"]["streams"] += 1
                report["validation"]["pictures"] += native.pictures+1
            cpu.close()
            print("PASS " + key, flush=True)
    if report["source_sha256"] != {name: digest((repository/name).read_bytes()) for name in sources}:
        raise RuntimeError("Source changed during export; regenerate with a fixed source revision")
    report["status"] = "pass"
    save_json(root/"bundle.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("codec_compute/artifacts/cpu-comparison"))
    parser.add_argument("--cpu-helper", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    result = prepare(args.output, cpu_helper=args.cpu_helper, sizes=tuple(args.sizes), jobs=args.jobs)
    print(json.dumps(result["validation"], indent=2))
