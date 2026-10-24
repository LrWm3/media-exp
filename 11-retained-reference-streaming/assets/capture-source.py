"""Retain four frozen 512px scenes and render exact saved decoder luma.

The complete original index is copied byte-for-byte as full-bundle.json. This
directory deliberately has no runnable bundle.json: omitted payloads remain
explicitly identified. No decoder, compiler, CPU kernel, or benchmark runs here.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SIZE = 512
COUNT = 32
FIXTURE = "scene"
WORKLOADS = ("patch-atlas", "motion-interpolation")
POLICIES = ("dpb16", "tight")
OUTPUTS = (0, 15, 31)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def record(root, path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(data), "sha256": digest(data)}


def records(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield value
        else:
            for child in value.values():
                yield from records(child)
    elif isinstance(value, list):
        for child in value:
            yield from records(child)


def inventory(items):
    result = {}
    for item in items:
        if item["path"] in result and result[item["path"]] != item:
            raise AssertionError("Conflicting artifact records: " + item["path"])
        result[item["path"]] = item
    return dict(sorted(result.items()))


def checked(root, item):
    relative = Path(item["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise AssertionError("Artifact path escapes its bundle")
    data = path.read_bytes()
    if len(data) != item["bytes"] or digest(data) != item["sha256"]:
        raise AssertionError("Artifact size or hash differs: " + item["path"])
    return data


def selection(full):
    wanted = {f"{size}/{workload}/{count}/{policy}"
              for size in (128, 512) for workload in WORKLOADS
              for count in (1, 8, 32) for policy in POLICIES}
    if (full.get("format") != "codec-compute-retained-inputs-1"
            or full.get("status") != "pass" or full.get("production_matrix") is not True
            or set(full.get("cases", {})) != wanted
            or set(full.get("decoders", ())) != {"ffmpeg", "openh264", "vaapi"}):
        raise AssertionError("Expected the complete passing three-decoder production export")
    chosen = {}
    for workload in WORKLOADS:
        for policy in POLICIES:
            key = f"{SIZE}/{workload}/{COUNT}/{policy}"
            case = full["cases"][key]
            jobs = [job for job in case["jobs"] if job["fixture"] == FIXTURE]
            if len(jobs) != 1:
                raise AssertionError("Expected exactly one scene fixture per retained case")
            chosen[key] = (case, jobs[0])
    return chosen


def selected_records(chosen):
    for case, job in chosen.values():
        yield from records({key: value for key, value in case.items() if key != "jobs"})
        yield from records(job)


def nal_types(data):
    marker = b"\x00\x00\x00\x01"
    if not data.startswith(marker):
        raise AssertionError("Expected four-byte Annex-B framing")
    parts = data.split(marker)[1:]
    if any(not part for part in parts):
        raise AssertionError("Empty NAL unit")
    return [part[0] & 31 for part in parts]


def audit_semantics(root, key, case):
    """Bind every generic CPU field to its public shared-IR output expression."""
    manifest = json.loads(checked(root, case["manifest"]))
    if (manifest["output_frames"] != case["output_frames"]
            or manifest["costs"] != case["costs"]
            or manifest["reference_capacity"] != case["reference_capacity"]
            or manifest["reference_slots"] != case["writer"]["reference_slots"]
            or manifest["prediction_slice_rows"] != 0
            or case["writer"]["prediction_slice_rows"] != 0):
        raise AssertionError("Compiler manifest differs from case metadata: " + key)
    graph = manifest["ir"]
    if graph["shape"] != [SIZE, SIZE]:
        raise AssertionError("Unexpected semantic image dimensions")
    nodes = {node["id"]: node for node in graph["nodes"]}
    visited = set()

    def node(identifier, operation, parameters, arity):
        item = nodes[identifier]
        if item["op"] != operation or item["params"] != parameters or len(item["args"]) != arity:
            raise AssertionError("Generic CPU fields differ from shared IR: " + key)
        visited.add(identifier)
        return item["args"]

    def image(identifier, name):
        node(identifier, "input", [name], 0)

    if [item["name"] for item in case["variants"]] != [f"output_{i:03}" for i in range(COUNT)]:
        raise AssertionError("Expected every output variant in its original order")
    expected_kind = 2 if case["workload"] == "patch-atlas" else 3
    for variant in case["variants"]:
        spec = variant["cpu"]
        if spec["kind"] != expected_kind or spec["rounds"] != 1:
            raise AssertionError("Unexpected generic CPU operation")
        for field_name, needed in (("field_a", True), ("field_b", expected_kind == 3)):
            field = spec[field_name]
            if (len(field) != ((SIZE // 16) ** 2 if needed else 0)
                    or any(len(pair) != 2 or any(type(value) is not int for value in pair) for pair in field)):
                raise AssertionError("Malformed generic CPU motion field")
        output = graph["outputs"][variant["name"]]
        if expected_kind == 2:
            parent, = node(output, "gather", [spec["field_a"], 16, "zero"], 1)
            image(parent, "image")
        else:
            first, second = node(output, "average", ["half_up"], 2)
            a, = node(first, "gather_h264", [spec["field_a"], 16], 1)
            b, = node(second, "gather_h264", [spec["field_b"], 16], 1)
            image(a, "a")
            image(b, "b")
    if visited != set(nodes) or set(graph["outputs"]) != set(case["output_frames"]):
        raise AssertionError("Unaccounted semantic nodes or named outputs")
    return manifest


def crop(frame, case):
    width = case["coded_width"]
    return b"".join(frame[(y + 16) * width + 16:(y + 16) * width + 16 + SIZE]
                    for y in range(SIZE))


def audit_case(root, key, case, job):
    manifest = audit_semantics(root, key, case)
    count = len(case["input_names"])
    width, height = case["coded_width"], case["coded_height"]
    if (case["size"] != SIZE or case["output_count"] != COUNT
            or (width, height) != (SIZE + 32, SIZE + 32)
            or case["frame_bytes"] != width * height * 3 // 2
            or case["costs"]["input_pictures"] != count
            or case["costs"]["computation_pictures"] != COUNT
            or case["costs"]["total_pictures"] != count + COUNT
            or case["costs"]["prediction_slices"] != COUNT
            or list(case["output_frames"].values()) != list(range(count, count + COUNT))):
        raise AssertionError("Unexpected retained sequence shape or picture counts")
    units = [checked(root, item) for item in job["units"]]
    predictions = [checked(root, item) for item in case["prediction_units"]]
    if (len(units) != count or len(predictions) != COUNT
            or nal_types(units[0]) != [7, 8, 5]
            or any(nal_types(unit) != [1] for unit in units[1:] + predictions)
            or checked(root, job["stream"]) != b"".join(units + predictions)
            or nal_types(checked(root, case["fence"])) != [7, 8, 5]):
        raise AssertionError("Ordinary stream framing or independent legacy fence differs")
    prediction_tail = b"".join(predictions)
    if (digest(prediction_tail) != manifest["prediction_sha256"]
            or len(prediction_tail) != case["costs"]["prediction_bytes"]):
        raise AssertionError("Frozen prediction tail differs from compiler manifest")
    raw = checked(root, job["expected"])
    frame_bytes = case["frame_bytes"]
    if len(raw) != (count + COUNT) * frame_bytes:
        raise AssertionError("Complete saved I420 picture count differs")
    expected = [job["inputs"][name] for name in case["input_names"]]
    expected += [job["outputs"][name] for name in case["output_frames"]]
    for index, compact_record in enumerate(expected):
        frame = raw[index * frame_bytes:(index + 1) * frame_bytes]
        if crop(frame, case) != checked(root, compact_record):
            raise AssertionError("Saved decoded crop differs from compact input/output")
        for y in range(height):
            row = frame[y * width:(y + 1) * width]
            if any(row if y < 16 or y >= SIZE + 16 else row[:16] + row[-16:]):
                raise AssertionError("Saved decoded luma guard is nonzero")
        if frame[width * height:] != bytes([128]) * (frame_bytes - width * height):
            raise AssertionError("Saved decoded chroma is not neutral")
    if set(job["decoder_validation"]) != {"ffmpeg", "openh264", "vaapi"}:
        raise AssertionError("Missing original local decoder validation")
    for validation in job["decoder_validation"].values():
        if (validation["exact"] is not True or validation["sha256"] != digest(raw)
                or validation["bytes"] != len(raw) or validation["pictures"] != count + COUNT):
            raise AssertionError("Original decoder validation differs from retained I420")
    return raw


def audit(root, *, include_capture=True):
    root = Path(root).resolve()
    if (root / "bundle.json").exists():
        raise AssertionError("Curated evidence must not advertise a runnable bundle.json")
    curation = json.loads((root / "curation.json").read_bytes())
    if curation.get("format") != "retained-inputs-curation-1" or curation.get("status") != "pass":
        raise AssertionError("Expected completed curation inventory")
    full = json.loads(checked(root, curation["full_bundle"]))
    chosen = selection(full)
    expected_files = inventory([curation["full_bundle"], curation["capture_script"], *selected_records(chosen)])
    if (curation["files"] != expected_files or curation["selected_cases"] != list(chosen)
            or curation["fixture"] != FIXTURE or curation["runnable_bundle"] is not False
            or curation["source_commit"] != full["source_commit"]
            or curation["original_validation"] != full["validation"]):
        raise AssertionError("Curation inventory differs from its exact full-index selection")
    all_files = inventory(records(full["cases"]))
    omitted = set(all_files) - set(inventory(selected_records(chosen)))
    omissions = curation["omissions"]
    if (omissions["omitted_cases"] != sorted(set(full["cases"]) - set(chosen))
            or omissions["omitted_fixtures_for_retained_cases"] != ["noise", "zero", "full"]
            or omissions["omitted_streams"] != sum(len(case["jobs"]) for case in full["cases"].values()) - len(chosen)
            or omissions["omitted_indexed_artifacts"] != len(omitted)
            or omissions["omitted_indexed_bytes"] != sum(all_files[path]["bytes"] for path in omitted)):
        raise AssertionError("Omission inventory differs from full-index selection")
    for item in expected_files.values():
        checked(root, item)
    raw_cases = {key: audit_case(root, key, case, job) for key, (case, job) in chosen.items()}
    for workload in WORKLOADS:
        first, second = (f"{SIZE}/{workload}/{COUNT}/{policy}" for policy in POLICIES)
        if raw_cases[first] != raw_cases[second]:
            raise AssertionError("Paired DPB declarations changed complete saved I420")
        if chosen[first][0]["variants"] != chosen[second][0]["variants"]:
            raise AssertionError("Paired DPB cases changed supplied CPU fields")
    result = {
        "retained_cases": len(chosen), "retained_streams": len(chosen),
        "retained_source_artifacts": len(expected_files),
        "retained_source_bytes": sum(item["bytes"] for item in expected_files.values()),
        "complete_i420_pictures_checked": sum(case["costs"]["total_pictures"] for case, _ in chosen.values()),
        "named_output_crops_checked": COUNT * len(chosen),
        "paired_complete_i420_matches": len(WORKLOADS),
        "full_bundle_sha256": curation["full_bundle"]["sha256"],
        "omitted_streams": omissions["omitted_streams"],
        "scope": "Saved export hashes, shared-IR/CPU field binding, framing, every compact crop, zero guards, neutral chroma and DPB-pair equality. No new decode or hardware timing.",
    }
    if include_capture:
        capture = json.loads((root / "capture.json").read_bytes())
        if (capture["curation"] != record(root, root / "curation.json")
                or capture["capture_script"] != curation["capture_script"]
                or capture["audits"] != result or len(capture["panels"]) != 18
                or len(capture["sheets"]) != len(WORKLOADS)):
            raise AssertionError("Capture provenance differs from audited curation")
        for panel in capture["panels"]:
            case, _ = chosen[panel["case"]]
            raw = raw_cases[panel["case"]]
            start = panel["frame_index"] * case["frame_bytes"]
            pixels = crop(raw[start:start + case["frame_bytes"]], case)
            checked(root, panel["panel"])
            with Image.open(root / panel["panel"]["path"]) as image:
                if image.mode != "L" or image.size != (SIZE, SIZE) or image.tobytes() != pixels:
                    raise AssertionError("Panel pixels differ from native decoded luma")
            if panel["plane_sha256"] != digest(pixels):
                raise AssertionError("Panel plane hash differs")
            if "sheet" in panel:
                sheet = capture["sheets"][panel["sheet"]]["image"]
                checked(root, sheet)
                left, top, width, height = panel["sheet_rectangle"]
                with Image.open(root / sheet["path"]) as image:
                    displayed = image.crop((left, top, left + width, top + height)).convert("L")
                    if displayed.size != (SIZE, SIZE) or displayed.tobytes() != pixels:
                        raise AssertionError("Contact sheet changed decoded luma pixels")
        result = {**result, "native_scale_panels_checked": len(capture["panels"]),
                  "contact_sheets_checked": len(capture["sheets"])}
    return result, chosen, raw_cases


def capture(root):
    audits, chosen, raw_cases = audit(root, include_capture=False)
    curation = json.loads((root / "curation.json").read_bytes())
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    heading = ImageFont.truetype("DejaVuSans.ttf", 28)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    evidence = {
        "format": "retained-inputs-capture-1", "capture_script": curation["capture_script"],
        "curation": record(root, root / "curation.json"), "full_bundle": curation["full_bundle"],
        "audits": audits, "source_commit": curation["source_commit"],
        "pixel_source": "Complete saved I420 checked against every compact output and original local decoder hashes.",
        "hardware_scope": "Images show frozen local FFmpeg/OpenH264/VAAPI evidence. This script performs no NVDEC run or timing.",
        "display": {"scale": 1, "interpolation": "none", "contrast_stretch": False,
                    "zero_luma": "black", "guards_and_chroma": "Audited; excluded from active-luma panels."},
        "panels": [], "sheets": [],
    }
    for workload in WORKLOADS:
        title = "Patch atlas" if workload == "patch-atlas" else "Motion interpolation"
        sheet = Image.new("RGB", (3 * (SIZE + 24) + 16, 2 * (SIZE + 66) + 146), "#101827")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 16), title + " · retained inputs", fill="white", font=heading)
        draw.text((20, 56), "512 × 512 · scene fixture · saved local decode · native 1:1 luma", fill="#bfd0df", font=font)
        sheet_index = len(evidence["sheets"])
        for row, policy in enumerate(POLICIES):
            key = f"{SIZE}/{workload}/{COUNT}/{policy}"
            case, job = chosen[key]
            raw = raw_cases[key]
            named = [("input-" + name, index, job["inputs"][name], None)
                     for index, name in enumerate(case["input_names"])]
            named += [(f"output-{index:02}", case["output_frames"][f"output_{index:03}"],
                       job["outputs"][f"output_{index:03}"], column)
                      for column, index in enumerate(OUTPUTS)]
            for label, frame_index, compact_record, column in named:
                start = frame_index * case["frame_bytes"]
                pixels = crop(raw[start:start + case["frame_bytes"]], case)
                panel = Image.frombytes("L", (SIZE, SIZE), pixels)
                path = root / "panels" / workload / policy / (label + ".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                panel.save(path)
                item = {"case": key, "fixture": FIXTURE, "label": label,
                        "frame_index": frame_index, "source_i420": job["expected"],
                        "compact_reference": compact_record, "i420_crop": [16, 16, SIZE, SIZE],
                        "plane_sha256": digest(pixels), "panel": record(root, path),
                        "pixel_min": min(pixels), "pixel_max": max(pixels),
                        "nonzero_pixels": sum(value != 0 for value in pixels)}
                if column is not None:
                    left, top = 20 + column * (SIZE + 24), 132 + row * (SIZE + 66)
                    capacity = case["reference_capacity"]
                    draw.text((left, top - 40), f"Output {OUTPUTS[column]} · {capacity} reference slots",
                              fill="white", font=font)
                    sheet.paste(panel.convert("RGB"), (left, top))
                    item.update(sheet=sheet_index, sheet_rectangle=[left, top, SIZE, SIZE])
                evidence["panels"].append(item)
        draw.text((20, sheet.height - 38), "Black = 0. No scaling or contrast changes. Inputs retained as separate native-size panels.",
                  fill="#bfd0df", font=small)
        path = root / (workload + "-outputs-512.png")
        sheet.save(path)
        evidence["sheets"].append({"workload": workload, "image": record(root, path),
                                   "columns": list(OUTPUTS), "rows": list(POLICIES)})
    write_json(root / "capture.json", evidence)
    return audit(root)[0]


def curate(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    source = bundle.parent
    if source == output or (output / "bundle.json").exists():
        raise ValueError("Choose a separate evidence directory without runnable bundle.json")
    full_bytes = bundle.read_bytes()
    full = json.loads(full_bytes)
    chosen = selection(full)
    files = inventory(selected_records(chosen))
    output.mkdir(parents=True, exist_ok=True)
    (output / "full-bundle.json").write_bytes(full_bytes)
    script = Path(__file__).resolve()
    (output / "capture-source.py").write_bytes(script.read_bytes())
    for item in files.values():
        data = checked(source, item)
        destination = output / item["path"]
        if not destination.resolve().is_relative_to(output):
            raise AssertionError("Destination artifact path escapes evidence directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    full_record = record(output, output / "full-bundle.json")
    script_record = record(output, output / "capture-source.py")
    all_files = inventory(records(full["cases"]))
    omitted = set(all_files) - set(files)
    retained = inventory([full_record, script_record, *files.values()])
    curation = {
        "format": "retained-inputs-curation-1", "status": "pass", "runnable_bundle": False,
        "full_bundle": full_record, "capture_script": script_record,
        "capture_script_repository_path": "11-retained-reference-streaming/curate.py",
        "source_commit": full["source_commit"], "original_validation": full["validation"],
        "selected_cases": list(chosen), "fixture": FIXTURE, "files": retained,
        "selection_policy": "Keep 512px, 32-output scene fixtures for atlas/interpolation under both DPB declarations; retain every artifact referenced by those four cases and fixtures.",
        "omissions": {
            "note": "full-bundle.json is the exact complete original index. Unselected payloads are intentionally absent. This directory is review evidence, not a runnable benchmark bundle.",
            "omitted_cases": sorted(set(full["cases"]) - set(chosen)),
            "omitted_fixtures_for_retained_cases": ["noise", "zero", "full"],
            "omitted_streams": sum(len(case["jobs"]) for case in full["cases"].values()) - len(chosen),
            "omitted_indexed_artifacts": len(omitted),
            "omitted_indexed_bytes": sum(all_files[path]["bytes"] for path in omitted),
        },
    }
    write_json(output / "curation.json", curation)
    return capture(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path, help="Complete frozen export bundle.json")
    mode.add_argument("--audit", type=Path, help="Previously curated evidence directory")
    parser.add_argument("--output", type=Path, help="Evidence destination when curating")
    args = parser.parse_args()
    if args.audit:
        result = audit(args.audit)[0]
    elif args.output:
        result = curate(args.bundle, args.output)
    else:
        parser.error("--output is required with --bundle")
    print(json.dumps(result, indent=2))
