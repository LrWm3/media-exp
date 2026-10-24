"""Freeze paired I420 references for slice layouts and host/completion interfaces."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import codec_compute as media
from decoder_core.decode import available, decode
from ..cpu_comparison.native import CPU
from ..cpu_comparison.tasks import chain_kernel
from ..resampling.kernels import Task, fixture, reference, tasks as resampling_tasks


LAYOUTS = {"rows-1": 1, "rows-4": 4, "full": 0}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tasks(size, depths):
    existing = {task.name: task for task in resampling_tasks(size)}
    for name, kind in (("control", 0), ("patch-atlas", 2), ("motion-interpolation", 3)):
        task = existing[name]
        yield task, {"kind": kind, "rounds": 1,
                     "field_a": task.fields.get("integer", task.fields.get("a_quarter", ())),
                     "field_b": task.fields.get("b_quarter", ())}
    first, second = ((1, 2),)*((size//16)**2), ((-1, -2),)*((size//16)**2)
    for depth in depths:
        task = Task("resident-chain-"+str(depth), "Resident warp/blend recurrence",
                    "Repeated rounded fractional resampling; only the final image is returned.",
                    ("image",), chain_kernel(depth, first, second), {"a_quarter": first, "b_quarter": second})
        yield task, {"kind": 4, "rounds": depth, "field_a": first, "field_b": second}


def prepare(output, *, cpu_helper, planes_helper, size=512, depths=(8, 32, 64)):
    if type(size) is not int or size not in (32, 128, 256, 512):
        raise ValueError("Supported sizes: 32 (smoke), 128, 256, 512")
    if (not depths or any(type(depth) is not int or not 1 <= depth <= 64 for depth in depths)
            or len(set(depths)) != len(depths)):
        raise ValueError("Distinct recurrence depths from 1 to 64 required")
    root, repository = Path(output).resolve(), Path(__file__).resolve().parents[3]
    root.mkdir(parents=True, exist_ok=True)
    sources = sorted(str(path.relative_to(repository))
                     for folder in ("codec_compute", "decoder_core", "05-h264-compiler/h264")
                     for path in (repository/folder).rglob("*")
                     if path.suffix in (".py", ".cpp", ".sh")
                     and not {"tests", "artifacts", "__pycache__", ".venv"}.intersection(path.parts))
    hashes = lambda: {name: digest((repository/name).read_bytes()) for name in sources}
    backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
    if "ffmpeg" not in backends:
        raise RuntimeError("FFmpeg is required")
    report = {"format": "codec-compute-resident-1", "experiment": "interface-limits", "status": "running",
              "sizes": [size], "depths": list(depths), "layouts": LAYOUTS, "jobs": 4,
              "fixtures": ["scene", "noise", "zero", "255"], "decoders": backends,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
              "source_sha256": hashes(), "cpu_helper_sha256": digest(Path(cpu_helper).read_bytes()),
              "planes_helper_sha256": digest(Path(planes_helper).read_bytes()), "cases": {},
              "validation": {"streams": 0, "pictures": 0, "local_decoder_runs": 0, "paired_complete_i420": 0}}

    def artifact(path, data):
        destination = root/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return {"path": str(path), "bytes": len(data), "sha256": digest(data)}

    level = {32: 31, 128: 31, 256: 41, 512: 51}[size]
    for task, cpu_spec in tasks(size, depths):
        helper = planes_helper if cpu_spec["kind"] == 4 else cpu_helper
        cpu = CPU(helper, size, cpu_spec)
        try:
            fixtures = []
            for index in range(4):
                images = fixture(task, index, size) if index < 2 else {
                    name: [[0 if index == 2 else 255]*size for _ in range(size)] for name in task.input_names}
                compact = {name: bytes(v for row in board for v in row) for name, board in images.items()}
                expected = cpu.run(*(compact[name] for name in task.input_names))
                if cpu_spec["kind"] != 4:
                    independent = reference(task, images)["output"]
                    if expected != bytes(v for row in independent for v in row):
                        raise AssertionError("Native CPU differs from independent scalar formula")
                fixtures.append((images, compact, expected))
            paired = {}
            for variant, rows in LAYOUTS.items():
                key = f"{size}/{task.name}/{variant}"
                program = media.compile(task.kernel, width=size, height=size, input_ranges=task.input_ranges,
                                        h264_level=level, h264_merge_zero_guards=True,
                                        h264_prediction_slice_rows=rows)
                manifest, native = program.manifest(), program.native
                writer = native.writer
                coded_w, coded_h = writer.width, writer.height
                frame_bytes = coded_w*coded_h*3//2
                case = {"size": size, "cpu": cpu_spec, "level_idc": level, "prediction_slice_rows": rows,
                        "input_names": list(task.input_names), "costs": manifest["costs"],
                        "coded_width": coded_w, "coded_height": coded_h, "frame_bytes": frame_bytes,
                        "output_frames": native.output_frames,
                        "writer": {"width": size, "height": size, "fps": writer.fps, "motion_bits": writer.motion_bits,
                                   "slice_mode": writer.slice_mode, "level_idc": level,
                                   "reference_slots": dict(writer.reference_slots), "merge_zero_guards": True,
                                   "prediction_slice_rows": rows},
                        "manifest": artifact(Path(key)/"manifest.json", (json.dumps(manifest, indent=2)+"\n").encode()),
                        "prediction_units": [artifact(Path(key)/f"prediction-{i:03}.h264", unit)
                                             for i, unit in enumerate(native.prediction_units)],
                        "fence": artifact(Path(key)/"fence.h264", native.fence_unit), "jobs": []}
                report["cases"][key] = case
                for index, (images, compact, expected) in enumerate(fixtures):
                    units = native.access_units(images)
                    ordinary = b"".join(units)
                    if ordinary != program.encode(images):
                        raise AssertionError("Framing changes compiled stream")
                    folder = Path(key)/f"{index:03}"
                    stream = artifact(folder/"program.h264", ordinary+native.fence_unit)
                    actual = None
                    for backend in backends:
                        raw = decode(root/stream["path"], backend)
                        if len(raw) != (native.pictures+1)*frame_bytes:
                            raise AssertionError("Decoded picture count differs")
                        if actual is not None and raw != actual:
                            raise AssertionError("Local decoders disagree on complete I420")
                        actual = raw
                        report["validation"]["local_decoder_runs"] += 1
                    frames = tuple(actual[i:i+frame_bytes] for i in range(0, len(actual), frame_bytes))
                    if frames[-1] != bytes([128])*frame_bytes:
                        raise AssertionError("Fence differs from constant 128")
                    for frame in frames[:-1]:
                        for y in range(coded_h):
                            row = frame[y*coded_w:(y+1)*coded_w]
                            if any(row if y < 16 or y >= size+16 else row[:16]+row[-16:]):
                                raise AssertionError("Nonzero computation guard")
                        if frame[coded_w*coded_h:] != bytes([128])*(frame_bytes-coded_w*coded_h):
                            raise AssertionError("Nonneutral computation chroma")
                    def plane(frame):
                        return b"".join(frame[(y+16)*coded_w+16:(y+16)*coded_w+16+size] for y in range(size))
                    for i, name in enumerate(task.input_names):
                        if plane(frames[i]) != compact[name]:
                            raise AssertionError("Decoded input differs")
                    if set(native.output_frames) != {"output"} or plane(frames[native.output_frames["output"]]) != expected:
                        raise AssertionError("Decoded named output differs from native CPU")
                    signature = (digest(actual), tuple(digest(unit) for unit in units[:len(task.input_names)]),
                                 native.output_frames, native.pictures, digest(native.fence_unit))
                    if index in paired:
                        if paired[index] != signature:
                            raise AssertionError("Slice layout changed paired input/pictures/complete I420")
                        report["validation"]["paired_complete_i420"] += 1
                    else:
                        paired[index] = signature
                    case["jobs"].append({"inputs": {name: artifact(folder/f"input-{name}.u8", data) for name, data in compact.items()},
                                         "units": [artifact(folder/f"input-{i:03}.h264", unit)
                                                   for i, unit in enumerate(units[:len(task.input_names)])],
                                         "stream": stream, "ordinary": artifact(folder/"ordinary.h264", ordinary),
                                         "expected": artifact(folder/"expected.i420", actual),
                                         "outputs": {"output": artifact(folder/"output.u8", expected)}})
                    report["validation"]["streams"] += 1
                    report["validation"]["pictures"] += native.pictures+1
                print("PASS "+key, flush=True)
        finally:
            cpu.close()
    if hashes() != report["source_sha256"]:
        raise RuntimeError("Source changed during export; regenerate after source freeze")
    report["status"] = "pass"
    (root/"bundle.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("codec_compute/artifacts/interface-limits"))
    parser.add_argument("--cpu-helper", type=Path, required=True)
    parser.add_argument("--planes-helper", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--depths", nargs="+", type=int, default=[8, 32, 64])
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, cpu_helper=args.cpu_helper, planes_helper=args.planes_helper,
                             size=args.size, depths=tuple(args.depths))["validation"], indent=2))
