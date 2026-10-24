"""Curate frozen fixtures and render their saved decoded luma without scaling.

The complete original index is preserved byte-for-byte. Its unselected payloads
are not copied. Panels come from the retained, hash-checked local decoded I420;
this script neither runs NVDEC nor infers a hardware benchmark pass.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..resident.benchmark import common, load


TASKS = (("control", "Copy control", 0, 1),
         ("average-control", "Rounded average", 1, 1),
         ("patch-atlas", "Patch atlas", 2, 1),
         ("motion-interpolation", "Fractional warp/blend", 3, 1),
         ("resident-chain-16", "Resident recurrence: 16", 4, 16),
         ("resident-chain-64", "Resident recurrence: 64", 4, 64))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(root, path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(root)), "bytes": len(data), "sha256": digest(data)}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+"\n")


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


def audit_cpu(root, key, case):
    """Bind native CPU kind, rounds and both motion fields to the semantic DAG."""
    expected = {name: (kind, rounds) for name, _, kind, rounds in TASKS}
    name = key.split("/")[1]
    spec = case["cpu"]
    if name not in expected or (spec["kind"], spec["rounds"]) != expected[name]:
        raise AssertionError("CPU task kind or round count differs: "+key)
    kind, rounds = spec["kind"], spec["rounds"]
    if type(kind) is not int or type(rounds) is not int:
        raise AssertionError("CPU task kind and round count must be integers")
    for field_name, needed in (("field_a", kind in (2, 3, 4)), ("field_b", kind in (3, 4))):
        field = spec[field_name]
        if len(field) != ((case["size"]//16)**2 if needed else 0):
            raise AssertionError("CPU motion grid size differs: "+key)
        if any(len(pair) != 2 or any(type(v) is not int or not -(2**31) <= v < 2**31 for v in pair)
               for pair in field):
            raise AssertionError("CPU motion grid requires signed integer pairs")
    graph = json.loads(common.checked(root, case["manifest"]))["ir"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    visited = set()

    def node(identifier, operation, parameters, arity):
        result = nodes[identifier]
        if result["op"] != operation or result["params"] != parameters or len(result["args"]) != arity:
            raise AssertionError("CPU specification differs from semantic graph: "+key)
        visited.add(identifier)
        return result["args"]

    def image(identifier, name):
        node(identifier, "input", [name], 0)

    def sample(identifier, field, integer=False):
        parameters = [spec[field], 16]+(["zero"] if integer else [])
        return node(identifier, "gather" if integer else "gather_h264", parameters, 1)[0]

    output = graph["outputs"]["output"]
    if kind == 0:
        image(output, "image")
    elif kind == 1:
        a, b = node(output, "average", ["half_up"], 2)
        image(a, "a")
        image(b, "b")
    elif kind == 2:
        image(sample(output, "field_a", integer=True), "image")
    elif kind == 3:
        a, b = node(output, "average", ["half_up"], 2)
        image(sample(a, "field_a"), "a")
        image(sample(b, "field_b"), "b")
    else:
        for _ in range(rounds):
            a, b = node(output, "average", ["half_up"], 2)
            a, b = sample(a, "field_a"), sample(b, "field_b")
            if a != b:
                raise AssertionError("CPU recurrence branches use different previous images")
            output = a
        image(output, "image")
    if visited != set(nodes) or set(graph["outputs"]) != {"output"}:
        raise AssertionError("Unaccounted semantic nodes or named outputs")


def audit(root):
    """Check framing, all file hashes, CPU semantics and exact subset provenance."""
    root = Path(root).resolve()
    bundle_bytes, bundle, workloads = load(root)
    if bundle.get("experiment") != "cpu-comparison" or bundle["jobs"] != 1:
        raise AssertionError("Expected a one-fixture CPU comparison selection")
    selection = bundle["selection"]
    full_bytes = common.checked(root, selection["full_bundle"])
    full = json.loads(full_bytes)
    if (full.get("status") != "pass" or full.get("experiment") != "cpu-comparison"
            or selection["original_jobs_per_case"] != full["jobs"]
            or selection["original_validation"] != full["validation"]):
        raise AssertionError("Full index or selection provenance differs")
    expected_keys = {f"{size}/{name}/fused" for size in (128, 256) for name, _, _, _ in TASKS}
    if set(workloads) != expected_keys or set(full["cases"]) != expected_keys:
        raise AssertionError("Expected all twelve size/task combinations")
    original = copy.deepcopy(full)
    original["jobs"] = 1
    original["selection"] = selection
    streams, pictures, files = 0, 0, {}
    for key, workload in workloads.items():
        case = workload["case"]
        index = selection["fixture_indices"][key]
        if type(index) is not int or not 0 <= index < full["jobs"]:
            raise AssertionError("Selected fixture index is invalid")
        original["cases"][key]["jobs"] = [full["cases"][key]["jobs"][index]]
        audit_cpu(root, key, case)
        job = case["jobs"][0]
        if common.checked(root, job["ordinary"]) != b"".join(workload["jobs"][0]["units"]):
            raise AssertionError("Ordinary stream differs from framed input/prediction units")
        for item in records(case):
            if item["path"] in files and files[item["path"]] != item:
                raise AssertionError("Conflicting artifact records")
            common.checked(root, item)
            files[item["path"]] = item
        streams += 1
        pictures += case["costs"]["total_pictures"]+1
    original["validation"] = {"streams": streams, "pictures": pictures,
                              "local_decoder_runs": streams*len(full["decoders"])}
    if bundle != original:
        raise AssertionError("Curated bundle changed original case data or provenance")
    return bundle_bytes, bundle, workloads, {
        "retained_artifacts": len(files), "retained_artifact_bytes": sum(item["bytes"] for item in files.values()),
        "ordinary_streams_checked": streams, "cpu_specs_bound_to_semantic_graphs": len(workloads),
        "full_bundle_sha256": digest(full_bytes),
        "scope": "Checks saved export evidence and exact subset identity; no new decoder run or hardware claim."}


def curate(source, output, fixture=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Curated destination must differ from the full export")
    full_bytes, full, _ = load(source)
    if full.get("experiment") != "cpu-comparison" or "selection" in full:
        raise ValueError("Expected original full CPU comparison export")
    if type(fixture) is not int or not 0 <= fixture < full["jobs"]:
        raise ValueError("Fixture must index an exported job")
    output.mkdir(parents=True, exist_ok=True)
    (output/"full-bundle.json").write_bytes(full_bytes)
    bundle = copy.deepcopy(full)
    bundle["jobs"] = 1
    for case in bundle["cases"].values():
        case["jobs"] = [case["jobs"][fixture]]
        for item in records(case):
            data = common.checked(source, item)
            destination = output/item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    streams = len(bundle["cases"])
    bundle["validation"] = {"streams": streams,
                            "pictures": sum(case["costs"]["total_pictures"]+1 for case in bundle["cases"].values()),
                            "local_decoder_runs": streams*len(bundle["decoders"])}
    bundle["selection"] = {
        "format": "codec-compute-fixture-selection-1",
        "full_bundle": record(output, output/"full-bundle.json"),
        "original_jobs_per_case": full["jobs"], "original_validation": full["validation"],
        "fixture_indices": {key: fixture for key in bundle["cases"]},
        "policy": "Keep the same fixture index for every case; retain every referenced artifact including ordinary streams.",
        "full_index_scope": "Byte-identical original index; payloads for unselected jobs are omitted here.",
        "validation_scope": "Retained subset of original passing local export; counts include the discarded IDR fence.",
        "benchmark_scope": "Hardware benchmark uses the complete original bundle, not this one-fixture selection."}
    write_json(output/"bundle.json", bundle)
    audit(output)
    return capture(output)


def capture(root):
    root = Path(root).resolve()
    bundle_bytes, bundle, workloads, audits = audit(root)
    title = ImageFont.truetype("DejaVuSans.ttf", 24)
    label = ImageFont.truetype("DejaVuSans.ttf", 15)
    detail = ImageFont.truetype("DejaVuSans.ttf", 12)
    script = Path(__file__).resolve()
    evidence = {"format": "codec-compute-cpu-reference-capture-1",
                "capture_script": {"path": str(script.relative_to(script.parents[3])),
                                   "sha256": digest(script.read_bytes())},
                "source_commit": bundle["source_commit"],
                "curated_bundle": record(root, root/"bundle.json"),
                "full_bundle": bundle["selection"]["full_bundle"], "audits": audits,
                "pixel_source": "Saved local decoded I420 from the frozen export; output crops match saved compact references.",
                "local_export_decoders": bundle["decoders"],
                "hardware_evidence": "This capture performs no hardware decode. Consult the separate complete benchmark report for NVDEC equality and timing.",
                "display": {"scale": 1, "interpolation": "none", "contrast_stretch": False,
                            "zero_luma": "black", "guards_and_chroma": "Audited, excluded from displayed active luma."},
                "sheets": [], "panels": []}
    for size in sorted(bundle["sizes"]):
        tile_width, tile_height = max(size, 232)+24, size+75
        image = Image.new("RGB", (3*tile_width+40, 2*tile_height+145), "#111827")
        draw = ImageDraw.Draw(image)
        draw.text((20, 15), f"Frozen decoder references · {size} × {size}", fill="white", font=title)
        fixture = bundle["selection"]["fixture_indices"][f"{size}/control/fused"]
        draw.text((20, 49), f"Fixture {fixture} · saved decoded output · native 1:1 pixels", fill="#b5c2d8", font=label)
        for column, (name, caption, _, _) in enumerate(TASKS):
            key = f"{size}/{name}/fused"
            w = workloads[key]
            case, job = w["case"], w["case"]["jobs"][0]
            frame_index = case["output_frames"]["output"]
            data = common.extract(w["jobs"][0]["frames"][frame_index], w["shape"])
            if data != common.checked(root, job["outputs"]["output"]):
                raise AssertionError("Displayed output differs from frozen compact reference")
            panel = Image.frombytes("L", (size, size), data)
            path = root/"panels"/str(size)/(name+".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(path)
            x = 20+(column % 3)*tile_width
            y = 86+(column//3)*tile_height
            draw.text((x, y), caption, fill="white", font=label)
            count = case["costs"]["computation_pictures"]
            unit = "picture" if count == 1 else "pictures"
            draw.text((x, y+21), f"{count} compute {unit} · output frame {frame_index}", fill="#8ed5bf", font=detail)
            left, top = x+(tile_width-24-size)//2, y+44
            image.paste(panel.convert("RGB"), (left, top))
            evidence["panels"].append({"case": key, "fixture": bundle["selection"]["fixture_indices"][key],
                                        "frame_index": frame_index, "source_i420": job["expected"],
                                        "compact_reference": job["outputs"]["output"],
                                        "plane_sha256": digest(data), "panel": record(root, path),
                                        "pixel_min": min(data), "pixel_max": max(data),
                                        "nonzero_pixels": sum(value != 0 for value in data),
                                        "i420_crop": [16, 16, size, size],
                                        "contact_sheet_rectangle": [left, top, size, size]})
        bottom = image.height-46
        draw.text((20, bottom), "No rescaling or contrast stretch. Black = zero. Active luma only; guards/chroma audited.",
                  fill="#b5c2d8", font=detail)
        draw.text((20, bottom+18), "Source: saved local decoder export. NVDEC validation and timings belong to the separate report.",
                  fill="#b5c2d8", font=detail)
        path = root/f"reference-outputs-{size}.png"
        image.save(path)
        evidence["sheets"].append({"size": size, "image": record(root, path)})
    if digest((root/"bundle.json").read_bytes()) != digest(bundle_bytes):
        raise AssertionError("Curated bundle changed during capture")
    write_json(root/"capture.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original full export directory")
    parser.add_argument("--output", type=Path, required=True, help="Curated asset directory")
    parser.add_argument("--fixture", type=int, default=0)
    args = parser.parse_args()
    result = curate(args.source, args.output, args.fixture)
    print(json.dumps(result["audits"], indent=2))
