#!/usr/bin/env python3
"""Retain representative frozen evidence; never create a runnable subset bundle.

The exact complete index stays in full-bundle.json. Payloads are retained only
for fixture 0 of atlas, interpolation and the longest recurrence, in row/full
layouts. Capture reads the complete original export before that export moves.
Neither curation nor capture constitutes a new decoder or hardware run.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
CAPTURE = REPOSITORY/"codec_compute/workloads/interface_limits/capture.py"
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
spec = importlib.util.spec_from_file_location("_interface_curator_capture", CAPTURE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
common = capture.runtime.common


def digest(data):
    return hashlib.sha256(data).hexdigest()


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


def record(root, path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(data), "sha256": digest(data)}


def selected_keys(bundle):
    if (bundle.get("experiment") != "interface-limits" or bundle.get("status") != "pass"
            or bundle.get("jobs") != 4 or bundle.get("sizes") != [512]
            or not bundle.get("depths")):
        raise ValueError("Expected the complete passing four-fixture 512 export")
    names = ("patch-atlas", "motion-interpolation", "resident-chain-"+str(max(bundle["depths"])))
    return ["512/"+name+"/"+layout for name in names for layout in ("rows-1", "full")]


def selected_records(bundle, keys):
    result = {}
    for key in keys:
        case = bundle["cases"][key]
        subset = {name: case[name] for name in ("manifest", "prediction_units", "fence")}
        subset["job"] = case["jobs"][0]
        for item in records(subset):
            if item["path"] in result and result[item["path"]] != item:
                raise AssertionError("Conflicting retained artifact records")
            result[item["path"]] = item
    return result


def audit(output):
    """Verify the curated payloads against the exact full index and capture."""
    from PIL import Image
    output = Path(output).resolve()
    if (output/"bundle.json").exists():
        raise AssertionError("Curated evidence must not look like a runnable bundle")
    curation = json.loads((output/"curation.json").read_text())
    full_data = common.checked(output, curation["full_bundle"])
    bundle = json.loads(full_data)
    keys = selected_keys(bundle)
    if (curation["selected_cases"] != keys or curation["selected_fixture_indices"] != {key: [0] for key in keys}
            or curation["full_bundle_sha256"] != digest(full_data)
            or curation["original_validation"] != bundle["validation"]
            or curation["source_commit"] != bundle["source_commit"]
            or curation["curation_script_sha256"] != digest(Path(__file__).read_bytes())):
        raise AssertionError("Selection or provenance differs from the full index")
    payloads = selected_records(bundle, keys)
    capture_record = curation["capture"]
    evidence = json.loads(common.checked(output, capture_record))
    capture_relative = str(CAPTURE.relative_to(REPOSITORY))
    if (evidence["bundle_sha256"] != digest(full_data)
            or evidence["source_commit"] != bundle["source_commit"]
            or evidence["script_sha256"] != digest(CAPTURE.read_bytes())
            or evidence["script_sha256"] != bundle["source_sha256"][capture_relative]
            or evidence["display"]["scale"] != 1 or evidence["display"]["contrast_stretch"] is not False):
        raise AssertionError("Capture is not bound to the frozen export/script/display policy")
    retained = {item["path"]: item for item in curation["retained_files"]}
    if len(retained) != len(curation["retained_files"]):
        raise AssertionError("Duplicate retained artifact record")
    generated = {curation["full_bundle"]["path"]: curation["full_bundle"], capture_record["path"]: capture_record}
    for item in evidence["panels"]+evidence["sheets"]:
        data = (output/item["path"]).read_bytes()
        generated[item["path"]] = {"path": item["path"], "bytes": len(data), "sha256": digest(data)}
        if generated[item["path"]]["sha256"] != item["sha256"]:
            raise AssertionError("Capture PNG differs from provenance")
    if retained != dict(payloads, **generated):
        raise AssertionError("Retained file set differs from selected payloads and capture")
    for item in retained.values():
        common.checked(output, item)
    if (curation["retained_file_count"] != len(retained)
            or curation["retained_bytes"] != sum(item["bytes"] for item in retained.values())
            or curation["retained_payload_count"] != len(payloads)):
        raise AssertionError("Retained artifact counts differ")
    selected_frames = {}
    for key in keys:
        case, job = bundle["cases"][key], bundle["cases"][key]["jobs"][0]
        manifest = json.loads(common.checked(output, case["manifest"]))
        shape = common.geometry(manifest)
        capture.runtime.audit_cpu(output, key, case)
        prefix = tuple(common.checked(output, item) for item in job["units"])
        tail = tuple(common.checked(output, item) for item in case["prediction_units"])
        fence = common.checked(output, case["fence"])
        ordinary = b"".join(prefix+tail)
        if (common.checked(output, job["ordinary"]) != ordinary
                or common.checked(output, job["stream"]) != ordinary+fence):
            raise AssertionError("Retained stream differs from framed units")
        raw = common.checked(output, job["expected"])
        size = shape["frame_bytes"]
        if len(raw) != (shape["pictures_per_job"]+1)*size or raw[-size:] != bytes([128])*size:
            raise AssertionError("Retained reference count or final fence differs")
        frames = tuple(raw[i:i+size] for i in range(0, len(raw)-size, size))
        common.validate_guards(frames, shape)
        for index, name in enumerate(case["input_names"]):
            if common.extract(frames[index], shape) != common.checked(output, job["inputs"][name]):
                raise AssertionError("Retained decoded input differs")
        for name, index in case["output_frames"].items():
            if common.extract(frames[index], shape) != common.checked(output, job["outputs"][name]):
                raise AssertionError("Retained named output differs")
        selected_frames[key] = (shape, frames)
    for left, right in zip(keys[::2], keys[1::2]):
        if selected_frames[left] != selected_frames[right]:
            raise AssertionError("Retained row/full complete I420 differs")
    expected_panels = {key for key in keys if "/patch-atlas/" not in key}
    if len(evidence["panels"]) != len(expected_panels) or {panel["case"] for panel in evidence["panels"]} != expected_panels:
        raise AssertionError("Capture selection differs from requested panels")
    for panel in evidence["panels"]:
        key = panel["case"]
        case, job = bundle["cases"][key], bundle["cases"][key]["jobs"][0]
        shape, frames = selected_frames[key]
        frame = frames[case["output_frames"]["output"]]
        compact = common.extract(frame, shape)
        if (panel["fixture"] != 0 or panel["frame"] != case["output_frames"]["output"]
                or panel["reference_i420"] != job["expected"] or panel["frame_sha256"] != digest(frame)
                or panel["active_luma_sha256"] != digest(compact)):
            raise AssertionError("Panel provenance differs from retained saved decoded I420")
        with Image.open(output/panel["path"]) as image:
            if image.mode != "L" or image.size != (shape["width"], shape["height"]) or image.tobytes() != compact:
                raise AssertionError("Panel pixels differ from saved decoded crop")
    return {"selected_cases": len(keys), "selected_jobs": len(keys), "retained_files": len(retained),
            "retained_bytes": sum(item["bytes"] for item in retained.values()),
            "full_bundle_sha256": digest(full_data), "panels": len(evidence["panels"])}


def curate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or (output/"bundle.json").exists():
        raise ValueError("Use a distinct evidence directory without bundle.json")
    full_data = (source/"bundle.json").read_bytes()
    bundle = json.loads(full_data)
    keys = selected_keys(bundle)
    payloads = selected_records(bundle, keys)
    output.mkdir(parents=True, exist_ok=True)
    for item in payloads.values():
        data = common.checked(source, item)
        destination = output/item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (output/"full-bundle.json").write_bytes(full_data)
    evidence = capture.capture(source, output)
    if (source/"bundle.json").read_bytes() != full_data:
        raise AssertionError("Full export index changed during capture")
    generated = [record(output, output/"full-bundle.json"), record(output, output/"capture.json")]
    generated += [record(output, output/item["path"]) for item in evidence["panels"]+evidence["sheets"]]
    retained = sorted(list(payloads.values())+generated, key=lambda item: item["path"])
    curation = {"format": "codec-compute-interface-curation-1", "source_commit": bundle["source_commit"],
                "curation_script_sha256": digest(Path(__file__).read_bytes()),
                "full_bundle": generated[0], "full_bundle_sha256": digest(full_data), "capture": generated[1],
                "selected_cases": keys, "selected_fixture_indices": {key: [0] for key in keys},
                "original_jobs_per_case": bundle["jobs"], "original_validation": bundle["validation"],
                "retained_payload_count": len(payloads), "retained_file_count": len(retained),
                "retained_bytes": sum(item["bytes"] for item in retained), "retained_files": retained,
                "selection_policy": "Fixture 0 of atlas, fractional interpolation and longest recurrence, with row-1/full layouts",
                "omitted_payloads": "Other cases and fixtures 1/2/3 appear in the exact full index but their payloads are not retained here",
                "execution_scope": "Evidence selection only. No bundle.json is written; this directory is not a runnable benchmark corpus",
                "validation_scope": "Hash/framing/CPU-graph/guard/chroma/paired-I420 audits of frozen local export; no fresh decoder or hardware run",
                "inventory_scope": "Selected payloads, full index and capture files only; excludes this self-referential curation index and independently supplied benchmark/probe reports"}
    (output/"curation.json").write_text(json.dumps(curation, indent=2)+"\n")
    return audit(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(curate(args.source, args.output), indent=2))
