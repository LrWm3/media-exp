"""Export paired, pixel-identical guard layouts over a recurrence-depth sweep."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import codec_compute as media
from decoder_core.decode import available, decode
from ..cpu_comparison.native import CPU
from ..cpu_comparison.tasks import chain_kernel
from ..resampling.kernels import Task, fixture


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(output, *, cpu_helper, size=512, depths=(1, 4, 8, 16, 32, 64)):
    if size not in (32, 128, 256, 512) or type(size) is not int:
        raise ValueError("Supported active sizes are 32, 128, 256 and 512")
    if not depths or len(set(depths)) != len(depths) or any(type(n) is not int or not 1 <= n <= 64 for n in depths):
        raise ValueError("Choose distinct integer recurrence depths in 1..64")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    sources = set()
    for folder in ("codec_compute", "decoder_core", "05-h264-compiler/h264"):
        for path in (repository/folder).rglob("*"):
            if (path.suffix in (".py", ".cpp", ".sh") and path.is_file()
                    and not {"artifacts", "tests", "__pycache__", ".venv"}.intersection(path.parts)):
                sources.add(str(path.relative_to(repository)))
    sources = sorted(sources)
    report = {"format": "codec-compute-resident-1", "experiment": "guard-fusion", "status": "running",
              "sizes": [size], "depths": list(depths), "jobs": 8,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
              "source_sha256": {name: digest((repository/name).read_bytes()) for name in sources},
              "cpu_helper_sha256": digest(Path(cpu_helper).read_bytes()), "cases": {},
              "validation": {"streams": 0, "pictures": 0, "local_decoder_runs": 0, "paired_i420": 0}}
    backends = [name for name in ("ffmpeg", "openh264", "vaapi") if available(name)]
    if "ffmpeg" not in backends:
        raise RuntimeError("FFmpeg is required")
    report["decoders"] = backends

    def artifact(path, data):
        destination = root/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return {"path": str(path), "bytes": len(data), "sha256": digest(data)}

    first, second = ((1, 2),)*((size//16)**2), ((-1, -2),)*((size//16)**2)
    level = {32: 31, 128: 31, 256: 41, 512: 51}[size]
    for rounds in depths:
        task = Task("resident-chain-"+str(rounds), "Resident warp/blend recurrence", "Exact fixed-field recurrence.",
                    ("image",), chain_kernel(rounds, first, second), {"a_quarter": first, "b_quarter": second})
        spec = {"kind": 4, "rounds": rounds, "field_a": first, "field_b": second}
        paired = []
        with CPU(cpu_helper, size, spec) as cpu:
            for variant, merged in (("baseline", False), ("merged", True)):
                key = "{}/{}/{}".format(size, task.name, variant)
                program = media.compile(task.kernel, width=size, height=size, input_ranges=task.input_ranges,
                                        h264_level=level, h264_merge_zero_guards=merged)
                manifest, native = program.manifest(), program.native
                writer = native.writer
                if manifest["costs"]["computation_pictures"] != rounds:
                    raise AssertionError("Unexpected recurrence picture count")
                if manifest["costs"]["prediction_slices"] != rounds*((size//16+2) if merged else (3*size//16+2)):
                    raise AssertionError("Unexpected prediction slice count")
                frame_bytes = writer.width*writer.height*3//2
                case = {"size": size, "cpu": spec, "input_names": ["image"], "level_idc": level,
                        "coded_width": writer.width, "coded_height": writer.height,
                        "frame_bytes": frame_bytes, "output_frames": native.output_frames, "costs": manifest["costs"],
                        "writer": {"width": size, "height": size, "fps": writer.fps, "motion_bits": writer.motion_bits,
                                   "slice_mode": writer.slice_mode, "level_idc": level,
                                   "reference_slots": dict(writer.reference_slots), "merge_zero_guards": merged},
                        "manifest": artifact(Path(key)/"manifest.json", (json.dumps(manifest, indent=2)+"\n").encode()),
                        "prediction_units": [artifact(Path(key)/"prediction-{:03}.h264".format(i), unit)
                                             for i, unit in enumerate(native.prediction_units)],
                        "fence": artifact(Path(key)/"fence.h264", native.fence_unit), "jobs": []}
                report["cases"][key] = case
                for index in range(8):
                    images = fixture(task, index, size)
                    if index in (2, 3):
                        images = {"image": [[255 if index == 3 else 0]*size for _ in range(size)]}
                    compact = bytes(value for row in images["image"] for value in row)
                    expected = cpu.run(compact)
                    units = native.access_units(images)
                    ordinary = b"".join(units)
                    if ordinary != program.encode(images):
                        raise AssertionError("Framed access units differ from ordinary program")
                    stream = ordinary+native.fence_unit
                    folder = Path(key)/"{:03}".format(index)
                    stream_record = artifact(folder/"program.h264", stream)
                    actual = None
                    for backend in backends:
                        raw = decode(root/stream_record["path"], backend)
                        if len(raw) != (native.pictures+1)*frame_bytes or (actual is not None and actual != raw):
                            raise AssertionError("Full decoder I420 differs")
                        actual = raw
                        report["validation"]["local_decoder_runs"] += 1
                    frames = tuple(actual[i:i+frame_bytes] for i in range(0, len(actual), frame_bytes))
                    if frames[-1] != bytes([128])*frame_bytes:
                        raise AssertionError("Completion fence differs")
                    for frame in frames[:-1]:
                        for y in range(writer.height):
                            row = frame[y*writer.width:(y+1)*writer.width]
                            if any(row if y < 16 or y >= size+16 else row[:16]+row[-16:]):
                                raise AssertionError("Nonzero computation guard")
                        if frame[writer.width*writer.height:] != bytes([128])*(frame_bytes-writer.width*writer.height):
                            raise AssertionError("Non-neutral chroma")

                    def plane(frame):
                        return b"".join(frame[(y+16)*writer.width+16:(y+16)*writer.width+16+size] for y in range(size))

                    if plane(frames[0]) != compact or plane(frames[native.output_frames["output"]]) != expected:
                        raise AssertionError("Decoded inputs/output differ from native CPU")
                    identity = (digest(actual), digest(units[0]), digest(compact), digest(expected))
                    if not merged:
                        paired.append(identity)
                    elif paired[index] != identity:
                        raise AssertionError("Guard merging changed paired decoded pictures")
                    else:
                        report["validation"]["paired_i420"] += 1
                    case["jobs"].append({"inputs": {"image": artifact(folder/"input-image.u8", compact)},
                                         "units": [artifact(folder/"input-000.h264", units[0])],
                                         "stream": stream_record, "ordinary": artifact(folder/"ordinary.h264", ordinary),
                                         "expected": artifact(folder/"expected.i420", actual),
                                         "outputs": {"output": artifact(folder/"output.u8", expected)}})
                    report["validation"]["streams"] += 1
                    report["validation"]["pictures"] += native.pictures+1
                print("PASS "+key, flush=True)
    if report["source_sha256"] != {name: digest((repository/name).read_bytes()) for name in sources}:
        raise RuntimeError("Sources changed during export; freeze and regenerate")
    report["status"] = "pass"
    (root/"bundle.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("codec_compute/artifacts/guard-fusion"))
    parser.add_argument("--cpu-helper", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    args = parser.parse_args()
    result = prepare(args.output, cpu_helper=args.cpu_helper, size=args.size, depths=tuple(args.depths))
    print(json.dumps(result["validation"], indent=2))
