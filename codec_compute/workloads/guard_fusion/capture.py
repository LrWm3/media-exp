"""Curate paired guard layouts and display saved decoded luma at native size.

The full export index remains byte-identical. This audits saved local evidence;
hardware correctness and performance belong to the separate benchmark report.
"""

import argparse
import copy
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from decoder_core.session import _unit
from ..cpu_comparison.capture import digest, record, records, write_json
from ..resident.benchmark import common, load


def expected_cases(bundle):
    sizes, depths = bundle.get("sizes"), bundle.get("depths")
    if (not isinstance(sizes, list) or not sizes or len(set(sizes)) != len(sizes)
            or any(type(size) is not int or size not in (32, 128, 256, 512) for size in sizes)):
        raise AssertionError("Expected distinct supported sizes: 32, 128, 256, 512")
    if (not isinstance(depths, list) or not depths or len(set(depths)) != len(depths)
            or any(type(depth) is not int or not 1 <= depth <= 64 for depth in depths)):
        raise AssertionError("Expected distinct integer depths in 1..64")
    return {f"{size}/resident-chain-{depth}/{variant}": (size, depth, variant == "merged")
            for size in sizes for depth in depths for variant in ("baseline", "merged")}


def counts(bundle):
    cases = expected_cases(bundle)
    if set(bundle["cases"]) != set(cases):
        raise AssertionError("Missing or extra size/depth/variant case")
    if type(bundle["jobs"]) is not int or bundle["jobs"] not in (1, 8):
        raise AssertionError("Expected eight original fixtures or one selected fixture")
    decoders = bundle["decoders"]
    if (not isinstance(decoders, list) or not decoders or len(set(decoders)) != len(decoders)
            or "ffmpeg" not in decoders or set(decoders)-{"ffmpeg", "openh264", "vaapi"}):
        raise AssertionError("Expected passing local decoder evidence")
    jobs = bundle["jobs"]
    return {"streams": len(cases)*jobs,
            "pictures": sum(depth+2 for _, depth, _ in cases.values())*jobs,
            "local_decoder_runs": len(cases)*jobs*len(decoders),
            "paired_i420": len(cases)//2*jobs}


def audit_graph(case, manifest, depth):
    """Bind the native CPU recurrence and static fields to every semantic node."""
    spec, size = case["cpu"], case["size"]
    if (type(spec["kind"]) is not int or spec["kind"] != 4
            or type(spec["rounds"]) is not int or spec["rounds"] != depth):
        raise AssertionError("CPU kind/rounds differ from recurrence case")
    for field, offset in (("field_a", [1, 2]), ("field_b", [-1, -2])):
        if (spec[field] != [offset]*((size//16)**2)
                or any(type(value) is not int for pair in spec[field] for value in pair)):
            raise AssertionError("Expected fixed quarter-pixel fields (+1,+2) and (-1,-2)")
    graph = manifest["ir"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]) or any(type(identifier) is not int for identifier in nodes):
        raise AssertionError("Semantic node identifiers are invalid or repeated")
    if (graph["shape"] != [size, size] or graph["dtype"] != "u8"
            or set(graph["outputs"]) != {"output"}):
        raise AssertionError("Unexpected semantic shape/type/output")
    visited = set()

    def node(identifier, operation, parameters, arity):
        found = nodes[identifier]
        if (type(identifier) is not int or found["op"] != operation or found["params"] != parameters
                or len(found["args"]) != arity or found["range"] != [0, 255]):
            raise AssertionError("CPU recurrence differs from semantic DAG")
        visited.add(identifier)
        return found["args"]

    output = graph["outputs"]["output"]
    for _ in range(depth):
        first, second = node(output, "average", ["half_up"], 2)
        first = node(first, "gather_h264", [spec["field_a"], 16], 1)[0]
        second = node(second, "gather_h264", [spec["field_b"], 16], 1)[0]
        if first != second:
            raise AssertionError("Recurrence sources have different previous images")
        output = first
    node(output, "input", ["image"], 0)
    if (visited != set(nodes)
            or graph["inputs"] != [{"name": "image", "id": output, "range": [0, 255]}]):
        raise AssertionError("Unexpected semantic inputs or unaccounted nodes")


def audit_payloads(root, bundle, workloads):
    cases = expected_cases(bundle)
    if set(workloads) != set(cases) or bundle["validation"] != counts(bundle):
        raise AssertionError("Case coverage or validation totals differ")
    files, manifests = {}, {}
    for key, (size, depth, merged) in cases.items():
        workload = workloads[key]
        case = workload["case"]
        manifest = json.loads(common.checked(root, case["manifest"]))
        manifests[key] = manifest
        slices = (size//16+2) if merged else (3*size//16+2)
        if (type(case["size"]) is not int or case["size"] != size
                or case["input_names"] != ["image"] or case["output_frames"] != {"output": depth}
                or case["level_idc"] != {32: 31, 128: 31, 256: 41, 512: 51}[size]
                or case["costs"]["input_pictures"] != 1 or case["costs"]["computation_pictures"] != depth
                or case["costs"]["total_pictures"] != depth+1
                or case["costs"]["prediction_slices"] != depth*slices):
            raise AssertionError("Recurrence case costs, shape or output differ: "+key)
        writer = case["writer"]
        if (writer["merge_zero_guards"] is not merged or manifest["merge_zero_guards"] is not merged
                or writer["slice_mode"] != "row" or writer["width"] != size or writer["height"] != size
                or writer["level_idc"] != case["level_idc"] or manifest["guard"] != 16
                or writer["reference_slots"] != manifest["reference_slots"]):
            raise AssertionError("Writer guard policy differs: "+key)
        if manifest["optimizations"]["zero_guard_merge"] != {
                "enabled": merged, "eligible_pictures": depth, "merged_pictures": depth if merged else 0}:
            raise AssertionError("Guard decision differs from expected recurrence")
        audit_graph(case, manifest, depth)
        if digest(b"".join(workload["tail"])) != manifest["prediction_sha256"]:
            raise AssertionError("Manifest prediction hash differs")
        for unit in workload["tail"]:
            _unit(unit, idr=False)
            if unit.count(b"\0\0\0\1") != slices:
                raise AssertionError("Actual prediction slice count differs")
        _unit(workload["fence"], idr=True, idr_id=1)
        for job, loaded in zip(case["jobs"], workload["jobs"]):
            _unit(loaded["prefix"][0], idr=True, idr_id=0)
            if common.checked(root, job["ordinary"]) != b"".join(loaded["units"]):
                raise AssertionError("Ordinary stream differs from framed units")
        for item in records(case):
            if item["path"] in files and files[item["path"]] != item:
                raise AssertionError("Conflicting artifact records")
            common.checked(root, item)
            files[item["path"]] = item
    for size in bundle["sizes"]:
        for depth in bundle["depths"]:
            prefix = f"{size}/resident-chain-{depth}/"
            baseline, merged = (workloads[prefix+name] for name in ("baseline", "merged"))
            if (baseline["case"]["cpu"] != merged["case"]["cpu"] or baseline["fence"] != merged["fence"]
                    or manifests[prefix+"baseline"]["ir"] != manifests[prefix+"merged"]["ir"]):
                raise AssertionError("Paired fields, graph or fence differs")
            for old, new in zip(baseline["jobs"], merged["jobs"]):
                if old["prefix"] != new["prefix"] or old["boards"] != new["boards"] or old["frames"] != new["frames"]:
                    raise AssertionError("Paired input or full decoded I420 differs")
    return {"retained_artifacts": len(files), "retained_artifact_bytes": sum(item["bytes"] for item in files.values()),
            "cpu_specs_bound_to_semantic_graphs": len(cases), "paired_i420_checked": bundle["validation"]["paired_i420"],
            "scope": "Hash, framing, guard/chroma, semantic-spec and paired-reference audit; no new decoder run."}


def audit(root):
    root = Path(root).resolve()
    data, bundle, workloads = load(root)
    if bundle.get("experiment") != "guard-fusion" or bundle["jobs"] != 1:
        raise AssertionError("Expected one-fixture guard-fusion selection")
    selection = bundle["selection"]
    full_bytes = common.checked(root, selection["full_bundle"])
    full = json.loads(full_bytes)
    if (full.get("experiment") != "guard-fusion" or full.get("status") != "pass" or full["jobs"] != 8
            or "selection" in full or full["validation"] != counts(full)
            or selection["original_validation"] != full["validation"]
            or selection["original_jobs_per_case"] != full["jobs"]
            or expected_cases(full) != expected_cases(bundle)
            or selection["fixture_indices"] != {key: 0 for key in full["cases"]}):
        raise AssertionError("Original index or subset provenance differs")
    original = copy.deepcopy(full)
    original["jobs"], original["selection"] = 1, selection
    for case in original["cases"].values():
        if len(case["jobs"]) != full["jobs"]:
            raise AssertionError("Original index is missing fixtures")
        case["jobs"] = case["jobs"][:1]
    original["validation"] = counts(original)
    if bundle != original:
        raise AssertionError("Curated selection changed original case data")
    checks = audit_payloads(root, bundle, workloads)
    checks["full_bundle_sha256"] = digest(full_bytes)
    return data, bundle, workloads, checks


def curate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Curated destination must differ from original export")
    full_bytes, full, workloads = load(source)
    if full.get("experiment") != "guard-fusion" or full["jobs"] != 8 or "selection" in full:
        raise ValueError("Expected original eight-fixture guard-fusion export")
    audit_payloads(source, full, workloads)
    output.mkdir(parents=True, exist_ok=True)
    (output/"full-bundle.json").write_bytes(full_bytes)
    bundle = copy.deepcopy(full)
    bundle["jobs"] = 1
    for case in bundle["cases"].values():
        case["jobs"] = case["jobs"][:1]
        for item in records(case):
            data = common.checked(source, item)
            path = output/item["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    bundle["validation"] = counts(bundle)
    bundle["selection"] = {
        "format": "codec-compute-fixture-selection-1", "full_bundle": record(output, output/"full-bundle.json"),
        "original_jobs_per_case": full["jobs"], "original_validation": full["validation"],
        "fixture_indices": {key: 0 for key in bundle["cases"]},
        "policy": "Fixture 0 for every declared size/depth and both variants; retain every referenced payload.",
        "full_index_scope": "Byte-identical full index; unselected fixture payloads omitted.",
        "validation_scope": "Selected original passing local evidence, including one discarded IDR fence per stream.",
        "benchmark_scope": "Separate hardware report uses the original full bundle."}
    write_json(output/"bundle.json", bundle)
    if (source/"bundle.json").read_bytes() != full_bytes:
        raise AssertionError("Source index changed during curation")
    return capture(output)


def capture(root):
    root = Path(root).resolve()
    data, bundle, workloads, checks = audit(root)
    script = Path(__file__).resolve()
    script_name = str(script.relative_to(script.parents[3]))
    script_hash = digest(script.read_bytes())
    pinned = bundle["source_sha256"].get(script_name)
    if pinned is not None and pinned != script_hash:
        raise AssertionError("Capture source differs from frozen export")
    title = ImageFont.truetype("DejaVuSans.ttf", 24)
    label = ImageFont.truetype("DejaVuSans.ttf", 15)
    detail = ImageFont.truetype("DejaVuSans.ttf", 12)
    evidence = {"format": "codec-compute-guard-fusion-capture-1", "source_commit": bundle["source_commit"],
                "capture_script": {"path": script_name, "sha256": script_hash, "pinned_in_export": pinned is not None},
                "curated_bundle": record(root, root/"bundle.json"), "full_bundle": bundle["selection"]["full_bundle"],
                "audits": checks, "local_export_decoders": bundle["decoders"],
                "pixel_source": "Saved local decoded I420 crops, checked against compact input/output planes.",
                "hardware_evidence": "No hardware decode or timing performed here; consult the separate benchmark report.",
                "display": {"scale": 1, "interpolation": "none", "contrast_stretch": False},
                "sheets": [], "panels": []}
    depth = max(bundle["depths"])
    for size in sorted(bundle["sizes"]):
        prefix = f"{size}/resident-chain-{depth}/"
        baseline, merged = (workloads[prefix+name] for name in ("baseline", "merged"))
        old_slices, new_slices = (w["case"]["costs"]["prediction_slices"] for w in (baseline, merged))
        tile = max(size, 250)+24
        sheet = Image.new("RGB", (3*tile+40, size+220), "#111827")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 15), f"Guard slice merging · {size} × {size} · depth {depth}", fill="white", font=title)
        draw.text((20, 50), f"Fixture 0 · {old_slices} → {new_slices} prediction slices · 0 differing output pixels",
                  fill="#b5c2d8", font=label)
        panels = (("input", "Input", baseline, 0, "Input upload unchanged"),
                  ("baseline", "Baseline output", baseline, depth, f"{old_slices} prediction slices"),
                  ("merged", "Merged output", merged, depth, f"{new_slices} prediction slices"))
        for column, (name, caption, workload, frame, description) in enumerate(panels):
            job = workload["case"]["jobs"][0]
            compact = job["inputs"]["image"] if name == "input" else job["outputs"]["output"]
            pixels = common.extract(workload["jobs"][0]["frames"][frame], workload["shape"])
            if pixels != common.checked(root, compact):
                raise AssertionError("Displayed crop differs from compact plane")
            panel = Image.frombytes("L", (size, size), pixels)
            path = root/"panels"/str(size)/(name+".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.save(path)
            x, y = 20+column*tile, 87
            draw.text((x, y), caption, fill="white", font=label)
            draw.text((x, y+22), description, fill="#8ed5bf", font=detail)
            left, top = x+(tile-24-size)//2, y+47
            sheet.paste(panel.convert("RGB"), (left, top))
            evidence["panels"].append({"size": size, "depth": depth, "role": name, "fixture": 0,
                                        "source_i420": job["expected"], "frame_index": frame,
                                        "compact_plane": compact, "plane_sha256": digest(pixels),
                                        "panel": record(root, path), "i420_crop": [16, 16, size, size],
                                        "contact_sheet_rectangle": [left, top, size, size]})
        draw.text((20, size+161), "Native 1:1 pixels. No resizing or contrast stretch. Active luma; zero guards and neutral chroma audited.",
                  fill="#b5c2d8", font=detail)
        draw.text((20, size+181), "Saved local decoder evidence. Hardware equality and timings belong to the separate report.",
                  fill="#b5c2d8", font=detail)
        path = root/f"guard-fusion-{size}.png"
        sheet.save(path)
        evidence["sheets"].append({"size": size, "depth": depth, "prediction_slices": [old_slices, new_slices],
                                   "differing_output_pixels": 0, "image": record(root, path)})
    if (root/"bundle.json").read_bytes() != data or digest(script.read_bytes()) != script_hash:
        raise AssertionError("Bundle or capture source changed while rendering")
    write_json(root/"capture.json", evidence)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original full guard-fusion export")
    parser.add_argument("--output", type=Path, required=True, help="Curated fixture-zero destination")
    args = parser.parse_args()
    result = curate(args.source, args.output)
    print(json.dumps(result["audits"], indent=2))
