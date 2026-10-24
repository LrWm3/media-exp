"""Freeze retained-input IR streams and independently verified decoder outputs.

Each bounded sequence uploads its original inputs once. Every variant becomes
one named output picture. The ordinary stream has one initial IDR and no fence;
an independent fence artifact is available for legacy resident comparisons.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess

import codec_compute as media
from decoder_core.decode import available, decode
from ..cpu_comparison.native import CPU
from ..resampling.kernels import fixture as existing_fixture, tasks as existing_tasks


WORKLOADS = ("patch-atlas", "motion-interpolation")
FIXTURES = ("scene", "noise", "zero", "full")
DECODERS = ("ffmpeg", "openh264", "vaapi")
DPB_POLICIES = ("dpb16", "tight")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def variants(size, workload, count):
    """Deterministic image-independent fields, shared verbatim with the CPU.

    Atlas phases rotate source tiles. Tiny smoke canvases exhaust tile phases
    sooner, then shift by an additional integer pixel to keep fields distinct.
    Interpolation phases move the two supplied fields in opposite directions.
    """
    if type(size) is not int or size not in (32, 128, 512):
        raise ValueError("Retained-input sizes are 32 (smoke), 128 and 512")
    if workload not in WORKLOADS:
        raise ValueError("Unknown retained-input workload")
    if type(count) is not int or count not in (1, 8, 32):
        raise ValueError("Output counts are 1, 8 and 32")
    columns = size // 16
    base = {task.name: task for task in existing_tasks(size)}[workload]
    result = []
    for index in range(count):
        if workload == "patch-atlas":
            phase_x, phase_y = index % columns, (index // columns) % columns
            extra = index // (columns * columns)
            first = tuple(
                (((2 * bx + by + phase_x) % columns) * 16 - bx * 16 + extra,
                 ((2 * by + bx + phase_y) % columns) * 16 - by * 16)
                if bx < columns // 2 and by < columns // 2 else (size - 1, size - 1)
                for by in range(columns) for bx in range(columns))
            second, kind = (), 2
        else:
            dx, dy = index % 8 - 4, index // 8 - 2
            first = tuple((x + dx, y + dy) for x, y in base.fields["a_quarter"])
            second = tuple((x - dx, y - dy) for x, y in base.fields["b_quarter"])
            kind = 3
        result.append({"name": f"output_{index:03}", "cpu": {
            "kind": kind, "rounds": 1, "field_a": first, "field_b": second}})
    if len({(item["cpu"]["field_a"], item["cpu"]["field_b"]) for item in result}) != count:
        raise AssertionError("Every output requires a distinct supplied field")
    return result


def compile_sequence(size, workload, count, *, dpb_policy="dpb16"):
    """Use public shared IR; no target-specific pixel-result construction."""
    specifications = variants(size, workload, count)
    if dpb_policy not in DPB_POLICIES:
        raise ValueError("Unknown DPB declaration policy")
    if workload == "patch-atlas":
        names = ("image",)

        def kernel(image):
            return {item["name"]: media.gather(image, item["cpu"]["field_a"])
                    for item in specifications}
    else:
        names = ("a", "b")

        def kernel(a, b):
            return {item["name"]: media.average(
                media.gather_h264(a, item["cpu"]["field_a"]),
                media.gather_h264(b, item["cpu"]["field_b"]))
                for item in specifications}
    program = media.compile(kernel, width=size, height=size,
                            input_ranges={name: (0, 255) for name in names},
                            h264_level=51 if size == 512 else 31,
                            h264_prediction_slice_rows=0,
                            **({"h264_tight_dpb": True} if dpb_policy == "tight" else {}))
    native = program.native
    if (len(native.operations) != count
            or list(native.output_frames.values()) != list(range(len(names), len(names) + count))
            or native.refreshes
            or any(tuple(source.frame for source in operation.sources) != tuple(range(len(names)))
                   for operation in native.operations)):
        raise AssertionError("Sequence must predict every variant directly from original inputs")
    if any(native.writer.slice_count(operation) != 1 for operation in native.operations):
        raise AssertionError("Every output must use one whole-picture prediction slice")
    if any(kind != 1 for unit in native.prediction_units for kind in nal_types(unit)):
        raise AssertionError("Prediction tail must contain only non-IDR prediction NALs")
    return program, specifications


def fixture(size, workload, label):
    if label not in FIXTURES:
        raise ValueError("Unknown retained-input fixture")
    task = {task.name: task for task in existing_tasks(size)}[workload]
    if label in ("zero", "full"):
        value = 255 if label == "full" else 0
        return {name: [[value] * size for _ in range(size)] for name in task.input_names}
    return existing_fixture(task, FIXTURES.index(label), size)


def compact(images):
    return {name: bytes(value for row in board for value in row) for name, board in images.items()}


def nal_types(unit):
    marker = b"\x00\x00\x00\x01"
    if not unit.startswith(marker):
        raise AssertionError("Expected four-byte Annex-B access units")
    pieces = unit.split(marker)[1:]
    if not pieces or any(not piece for piece in pieces):
        raise AssertionError("Malformed Annex-B access unit")
    return [piece[0] & 31 for piece in pieces]


def full_frame(active, size):
    """Independent I420 expectation, including every guard and chroma byte."""
    if type(active) is not bytes or len(active) != size * size:
        raise ValueError("Compact luma must contain exactly size squared bytes")
    coded = size + 32
    luma = bytearray(coded * coded)
    for y in range(size):
        offset = (y + 16) * coded + 16
        luma[offset:offset + size] = active[y * size:(y + 1) * size]
    return bytes(luma) + bytes([128]) * (coded * coded // 2)


def source_snapshot(repository):
    roots = (repository / "codec_compute", repository / "decoder_core",
             repository / "05-h264-compiler" / "h264")
    suffixes = {".py", ".cpp", ".h", ".hpp", ".sh"}
    paths = sorted(path for root in roots for path in root.rglob("*")
                   if path.is_file() and path.suffix in suffixes
                   and not any(part in ("artifacts", "assets", ".build", "__pycache__")
                               for part in path.relative_to(repository).parts))
    return {str(path.relative_to(repository)): digest(path.read_bytes()) for path in paths}


def prepare(output, *, cpu_helper, sizes=(128, 512), output_counts=(1, 8, 32),
            decoders=DECODERS, dpb_policies=DPB_POLICIES):
    """Export defaults as the full matrix; explicit 32px runs are smoke only."""
    if not sizes or len(set(sizes)) != len(sizes) or any(type(size) is not int or size not in (32, 128, 512) for size in sizes):
        raise ValueError("Supply unique supported sizes")
    if (not output_counts or len(set(output_counts)) != len(output_counts)
            or any(type(count) is not int or count not in (1, 8, 32) for count in output_counts)):
        raise ValueError("Supply unique output counts from 1, 8 and 32")
    if (not decoders or len(set(decoders)) != len(decoders)
            or any(backend not in DECODERS for backend in decoders)):
        raise ValueError("Supply unique local decoder names")
    if (not dpb_policies or len(set(dpb_policies)) != len(dpb_policies)
            or any(policy not in DPB_POLICIES for policy in dpb_policies)):
        raise ValueError("Supply unique DPB declaration policies")
    missing = [backend for backend in decoders if not available(backend)]
    if missing:
        raise RuntimeError("Required local decoders are unavailable: " + ", ".join(missing))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    helper = Path(cpu_helper).resolve()
    sources = source_snapshot(repository)
    helper_digest = digest(helper.read_bytes())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    report = {
        "format": "codec-compute-retained-inputs-1", "status": "running",
        "experiment": "retained-inputs", "sizes": list(sizes),
        "output_counts": list(output_counts), "fixtures": list(FIXTURES),
        "decoders": list(decoders), "dpb_policies": list(dpb_policies),
        "production_matrix": (set(sizes) == {128, 512} and set(output_counts) == {1, 8, 32}
                              and set(decoders) == set(DECODERS) and set(dpb_policies) == set(DPB_POLICIES)),
        "source_commit": commit, "export_commit": commit, "source_sha256": sources,
        "cpu_helper_sha256": helper_digest,
        "runtime_contract": "One bounded sequence: upload inputs once, emit every named output, then stop. No unplanned append after final input-slot reuse.",
        "cpu_contract": "Generic C++ CPU kind 2/3, one exact supplied-field execution per variant. CPU results are validation artifacts only; stream encoding receives original inputs.",
        "cases": {}, "validation": {"streams": 0, "pictures": 0, "local_decoder_runs": 0,
                                     "named_outputs": 0, "retained_beyond_16_cases": 0,
                                     "paired_policy_outputs": 0},
    }
    # An interrupted replacement must not leave a previously passing manifest
    # advertising partially overwritten files as a verified frozen bundle.
    (root / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")

    def artifact(relative, data):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path.relative_to(root)), "bytes": len(data), "sha256": digest(data)}

    paired = {}
    for size in sizes:
        for workload in WORKLOADS:
            for count, policy in product(output_counts, dpb_policies):
                program, specifications = compile_sequence(size, workload, count, dpb_policy=policy)
                native, writer = program.native, program.native.writer
                names = [node.params[0] for node in program.graph.inputs]
                manifest = program.manifest()
                key = f"{size}/{workload}/{count}/{policy}"
                folder = Path(key)
                maximum_age = max(frame - source.frame
                                  for frame, operation in enumerate(native.operations, len(names))
                                  for source in operation.sources)
                if count == 32 and maximum_age <= 16:
                    raise AssertionError("32-output sequence must reuse inputs beyond sliding DPB age")
                case = {
                    "size": size, "workload": workload, "output_count": count,
                    "dpb_policy": policy, "reference_capacity": writer.reference_capacity,
                    "input_names": names, "level_idc": writer.level_idc,
                    "coded_width": writer.width, "coded_height": writer.height,
                    "frame_bytes": writer.width * writer.height * 3 // 2,
                    "costs": manifest["costs"], "output_frames": native.output_frames,
                    "variants": specifications, "maximum_input_reference_age": maximum_age,
                    "retained_beyond_16": maximum_age > 16,
                    "writer": {"width": size, "height": size, "fps": writer.fps,
                               "motion_bits": writer.motion_bits, "slice_mode": writer.slice_mode,
                               "level_idc": writer.level_idc, "reference_slots": dict(writer.reference_slots),
                               "reference_capacity": writer.reference_capacity,
                               "merge_zero_guards": writer.merge_zero_guards,
                               "prediction_slice_rows": writer.prediction_slice_rows},
                    "manifest": artifact(folder / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode()),
                    "prediction_units": [artifact(folder / f"prediction-{index:03}.h264", unit)
                                         for index, unit in enumerate(native.prediction_units)],
                    "fence": artifact(folder / "legacy-fence.h264", native.fence_unit),
                    "fence_policy": "Optional legacy runtime artifact; absent from every ordinary stream and expected I420.",
                    "jobs": [],
                }
                report["cases"][key] = case
                report["validation"]["retained_beyond_16_cases"] += int(maximum_age > 16)
                with ExitStack() as stack:
                    processors = [stack.enter_context(CPU(helper, size, item["cpu"]))
                                  for item in specifications]
                    for label in FIXTURES:
                        images = fixture(size, workload, label)
                        inputs = compact(images)
                        expected = {item["name"]: cpu.run(*(inputs[name] for name in names))
                                    for item, cpu in zip(specifications, processors)}
                        units = native.access_units(images)
                        stream = b"".join(units)
                        if stream != program.encode(images):
                            raise AssertionError("Access-unit framing changes the ordinary stream")
                        nals = Counter(nal_types(stream))
                        if nals[5] != 1 or nals[7] != 1 or nals[8] != 1 or set(nals) != {1, 5, 7, 8}:
                            raise AssertionError("Ordinary sequence requires exactly one initial IDR and parameter-set pair")
                        expected_i420 = b"".join(full_frame(value, size) for value in
                                                 [*(inputs[name] for name in names), *expected.values()])
                        job_folder = folder / label
                        stream_record = artifact(job_folder / "program.h264", stream)
                        checks = {}
                        for backend in decoders:
                            raw = decode(root / stream_record["path"], backend)
                            if raw != expected_i420:
                                raise AssertionError(f"{key}/{label}/{backend}: complete I420 differs from independent CPU/input, guard or chroma expectation")
                            checks[backend] = {"bytes": len(raw), "sha256": digest(raw),
                                               "pictures": native.pictures, "exact": True}
                            report["validation"]["local_decoder_runs"] += 1
                        pair_key = (size, workload, count, label)
                        complete_digest = digest(expected_i420)
                        if pair_key in paired:
                            if paired[pair_key] != complete_digest:
                                raise AssertionError("DPB declaration policies changed complete decoded I420")
                            report["validation"]["paired_policy_outputs"] += count
                        else:
                            paired[pair_key] = complete_digest
                        outputs = {name: artifact(job_folder / f"{name}.u8", value)
                                   for name, value in expected.items()}
                        if label == "noise" and len(set(expected.values())) != count:
                            raise AssertionError("Noise fixture must make every supplied-field output distinct")
                        case["jobs"].append({
                            "fixture": label,
                            "inputs": {name: artifact(job_folder / f"input-{name}.u8", value)
                                       for name, value in inputs.items()},
                            "units": [artifact(job_folder / f"input-{index:03}.h264", unit)
                                      for index, unit in enumerate(units[:len(names)])],
                            "stream": stream_record,
                            "expected": artifact(job_folder / "expected.i420", expected_i420),
                            "outputs": outputs, "decoder_validation": checks,
                            "distinct_output_count": len(set(expected.values())),
                            "nal_counts": dict(sorted(nals.items())), "idr_pictures": 1,
                        })
                        report["validation"]["streams"] += 1
                        report["validation"]["pictures"] += native.pictures
                        report["validation"]["named_outputs"] += count
                print("PASS " + key, flush=True)
    if sources != source_snapshot(repository) or helper_digest != digest(helper.read_bytes()):
        raise RuntimeError("Source or CPU helper changed during export; regenerate after freezing runtime source")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip() != commit:
        raise RuntimeError("Source commit changed during export; regenerate from one fixed revision")
    report["status"] = "pass"
    (root / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-helper", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--output-counts", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--decoders", nargs="+", choices=DECODERS, default=list(DECODERS))
    parser.add_argument("--dpb-policies", nargs="+", choices=DPB_POLICIES, default=list(DPB_POLICIES))
    args = parser.parse_args()
    result = prepare(args.output, cpu_helper=args.cpu_helper, sizes=tuple(args.sizes),
                     output_counts=tuple(args.output_counts), decoders=tuple(args.decoders),
                     dpb_policies=tuple(args.dpb_policies))
    print(json.dumps(result["validation"], indent=2))
