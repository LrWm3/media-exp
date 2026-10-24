#!/usr/bin/env python3
"""Bounded construction/schedule comparison for instruction-optimal parity.

Six explicitly listed candidates are measured, not an exhaustive joint search.
Each construction uses the proved minimum of three scalar instructions. The
serial construction has depth three; the parallel construction has depth two.
Their emitted H.264 costs expose a picture-count versus coded-area tradeoff.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess

from atlas import INPUTS, build_atlas, evaluate_plan
from program import Program
from decoder_core.decode import available, decode


ROOT = Path(__file__).resolve().parent
METRICS = ("prediction_pictures", "prediction_coded_luma_samples",
           "prediction_slices", "prediction_bytes", "reference_slots_used")
SCOPE = (
    "Bounded comparison of two hand-specified scalar-minimum constructions and "
    "three construction/schedule combinations per function. Candidate enumeration "
    "is not exhaustive; Pareto results apply only to these candidates and this "
    "geometry, writer, and retained-reference policy. No general joint optimizer "
    "or global optimum for bytes, slices, area, or reference slots is claimed."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_bytes(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(relative), "sha256": digest(data), "bytes": len(data)}


def _step(inputs, weights, bias=0):
    return {"inputs": inputs, "weights": weights, "bias": bias, "denominator": 0}


def dependency_layers(plan):
    """Earliest legal pictures for this fixed DAG; inputs are already resident."""
    depths, layers = [0, 0], []
    for result, step in enumerate(plan, 2):
        depth = 1 + max(depths[source] for source in step["inputs"])
        depths.append(depth)
        while len(layers) < depth:
            layers.append([])
        layers[depth - 1].append(result)
    return layers


def build_candidates(atlas=None):
    """Return the six declared candidates, retaining function-global minima."""
    atlas = build_atlas() if atlas is None else atlas
    functions = {case["id"]: case for case in atlas["functions"]}
    candidates = []
    for mask in (6, 9):
        original = functions[mask]
        for construction, packed in (("parallel", True), ("parallel", False), ("serial", True)):
            case = deepcopy(original)
            if construction == "serial":
                final_weights, final_bias = ([2, -4], 0) if mask == 6 else ([-2, 4], 1)
                case["plan"] = [_step([0, 1], [2, 2]), _step([2], [1], -1),
                                _step([2, 3], final_weights, final_bias)]
                case["output_index"] = 4
            layers = dependency_layers(case["plan"])
            # This schedule describes the supplied DAG. The function-global
            # packed_minimum remains two, which this serial DAG cannot attain.
            case["packed_schedule"] = layers
            actual = [evaluate_plan(case["plan"], values, case["output_index"]) for values in INPUTS]
            if actual != case["truth"] or len(case["plan"]) != case["scalar_minimum"]:
                raise AssertionError("Candidate is not a correct instruction-optimal witness")
            identifier = "%s-%s-%s" % (case["name"].lower(), construction, "packed" if packed else "unpacked")
            candidates.append({"id": identifier, "case": case, "construction": construction,
                               "packed": packed, "dependency_layers": layers,
                               "dependency_depth": len(layers), "truth_oracle": actual})
    return candidates


def pareto_frontier(records, metrics=METRICS):
    """Return nondominated IDs; lower is better for every requested metric."""
    frontier = []
    for candidate in records:
        values = candidate["costs"]
        dominated = any(
            other["id"] != candidate["id"]
            and all(other["costs"][key] <= values[key] for key in metrics)
            and any(other["costs"][key] < values[key] for key in metrics)
            for other in records
        )
        if not dominated:
            frontier.append(candidate["id"])
    return frontier


def input_patterns(width, height):
    pattern = {"a": [[((x + y) % 4) // 2 for x in range(width)] for y in range(height)],
               "b": [[(x + y) % 2 for x in range(width)] for y in range(height)]}
    rng = random.Random(808080)
    randomized = {name: [[rng.randrange(2) for _ in range(width)] for _ in range(height)]
                  for name in ("a", "b")}
    # Both fixtures include every truth pair explicitly, including at borders.
    for index, (a, b) in enumerate(INPUTS):
        randomized["a"][0][index], randomized["b"][0][index] = a, b
    return (("truth-pattern", pattern), ("seeded-random", randomized))


def plane(board):
    return bytes(value for row in board for value in row)


def _validate_inputs(program, raw, inputs):
    stride = program.writer.width * program.writer.height * 3 // 2
    for frame, name in enumerate(("a", "b")):
        values = b"".join(raw[frame * stride + (row + 16) * program.writer.width + 16:
                               frame * stride + (row + 16) * program.writer.width + 16 + program.width]
                          for row in range(program.height))
        if values != plane(inputs[name]):
            raise AssertionError("Decoder changed a literal input plane")


def prepare_joint(output, decoders=("ffmpeg", "openh264", "vaapi"), *, width=32, height=32):
    """Emit candidates, validate real decoders, and save a portable comparison."""
    output = Path(output).resolve()
    if not available("ffmpeg"):
        raise RuntimeError("FFmpeg is required for full I420 reference generation")
    requested = tuple(dict.fromkeys(("ffmpeg",) + tuple(decoders)))
    if any(name not in ("ffmpeg", "openh264", "vaapi") for name in requested):
        raise ValueError("Supported validation decoders: ffmpeg, openh264, vaapi")
    ready = tuple(name for name in requested if available(name))
    atlas = build_atlas()
    candidates = build_candidates(atlas)
    # Validate geometry before creating any fixtures or output files.
    programs = {item["id"]: Program(item["case"], width, height, packed=item["packed"])
                for item in candidates}
    write_json(output / "atlas.json", atlas)
    proofs = save_bytes(output, Path("PROOFS.md"), (ROOT / "PROOFS.md").read_bytes())
    certificate = atlas["single_instruction_certificate"]
    model = {"format": "h264-compute-limits-joint-model-0.1", "scope": SCOPE,
             "sample_domain": [0, 1], "intermediate_domain": [0, 255],
             "rounds": 1, "source_model": atlas["model"],
             "scalar_minimum": "three instructions for XOR/XNOR, proved in PROOFS.md",
             "fixed_dag_schedule_lower_bound": "each dependency edge crosses a picture; depth lower-bounds picture count for this fixed construction",
             "truth_order": ["00", "01", "10", "11"],
             "candidate_count": 6, "constructions_per_function": 2,
             "schedule_combinations_per_function": 3}
    write_json(output / "model.json", model)
    source_paths = [ROOT / name for name in ("joint.py", "test_joint.py", "atlas.py", "program.py", "PROOFS.md")]
    source_paths.extend(ROOT.parent / "decoder_core" / name for name in ("bitstream.py", "decode.py"))
    source_hashes = {str(path.relative_to(ROOT.parent)): digest(path.read_bytes()) for path in source_paths}
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                  "source_sha256": source_hashes,
                  "atlas_sha256": digest((output / "atlas.json").read_bytes()),
                  "certificate_sha256": certificate["sha256"]}
    write_json(output / "provenance.json", provenance)
    report = {"format": "h264-compute-limits-bundle-0.1", "experiment": "bounded-joint-candidate-comparison",
              "created_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
              "status": "pass", "width": width, "height": height,
              "atlas": "atlas.json", "atlas_sha256": provenance["atlas_sha256"],
              "certificate_sha256": certificate["sha256"],
              "model": "model.json", "model_sha256": digest((output / "model.json").read_bytes()),
              "proofs": proofs["path"], "proofs_sha256": proofs["sha256"],
              "provenance": "provenance.json", "provenance_sha256": digest((output / "provenance.json").read_bytes()),
              "source_commit": provenance["source_commit"], "source_sha256": source_hashes,
              "working_tree_dirty": provenance["working_tree_dirty"],
              "decoders": {name: "available" if name in ready else "unavailable" for name in requested},
              "input_seed": 808080, "decision_metrics": list(METRICS),
              "metric_meaning": {"prediction_pictures": "computation pictures after two input pictures",
                                 "prediction_coded_luma_samples": "coded width times height times prediction pictures, including guards/unused tiles",
                                 "coded_luma_samples": "all pictures including the two literal input pictures",
                                 "prediction_slices": "actual VCL prediction slices from the writer",
                                 "prediction_bytes": "actual encoded immutable prediction tail bytes",
                                 "reference_slots_used": "distinct retained-reference slots used by this emitted stream"},
              "cases": [], "decisions": []}
    fixtures = input_patterns(width, height)
    for candidate in candidates:
        case, identifier = candidate["case"], candidate["id"]
        program = programs[identifier]
        directory = Path("cases") / identifier
        manifest = program.manifest()
        manifest["joint_candidate"] = {"construction": candidate["construction"],
                                       "function_global_minima": {"scalar": case["scalar_minimum"], "packed": case["packed_minimum"]},
                                       "fixed_construction_depth": candidate["dependency_depth"],
                                       "scope": SCOPE}
        write_json(output / directory / "program.json", manifest)
        tail = save_bytes(output, directory / "prediction.h264", program.tail)
        fixture_records = []
        for label, inputs in fixtures:
            location = directory / label
            prefix = program.encode_inputs(inputs)
            stream = save_bytes(output, location / "program.h264", prefix + program.tail)
            expected = program.reference(inputs)
            raw = decode(output / stream["path"], "ffmpeg")
            if program.read(raw) != expected:
                raise AssertionError(identifier + ": truth-table oracle mismatch")
            _validate_inputs(program, raw, inputs)
            checks = {}
            for backend in ready:
                decoded = raw if backend == "ffmpeg" else decode(output / stream["path"], backend)
                if decoded != raw or program.read(decoded) != expected:
                    raise AssertionError(identifier + ": decoder differs: " + backend)
                checks[backend] = {"status": "pass", "full_i420_bytes": len(decoded),
                                   "full_planes_compared": True, "truth_output_compared": True}
            expected_raw = save_bytes(output, location / "expected.i420", raw)
            expected_output = save_bytes(output, location / "output.u8", plane(expected["output"]))
            saved_inputs = {name: save_bytes(output, location / (name + ".u8"), plane(board))
                            for name, board in inputs.items()}
            fixture_records.append({"label": label, "input_bytes": len(prefix),
                                    "stream": stream["path"], "stream_sha256": stream["sha256"], "stream_bytes": stream["bytes"],
                                    "expected": expected_raw["path"], "expected_sha256": expected_raw["sha256"], "expected_bytes": expected_raw["bytes"],
                                    "expected_output": expected_output["path"], "expected_output_sha256": expected_output["sha256"],
                                    "expected_output_bytes": expected_output["bytes"], "inputs": saved_inputs,
                                    "truth_pairs_present": [list(values) for values in INPUTS], "decoders": checks})
        costs = dict(program.costs)
        costs["prediction_coded_luma_samples"] = program.writer.width * program.writer.height * len(program.operations)
        record = {"id": identifier, "function": case["name"], "mask": case["id"],
                  "construction": candidate["construction"], "mode": "packed" if candidate["packed"] else "scalar",
                  "purpose": "joint", "rounds": 1, "evaluations": width * height,
                  "width": width, "height": height, "coded_width": program.writer.width,
                  "coded_height": program.writer.height, "pictures": program.pictures,
                  "frame_bytes": program.writer.width * program.writer.height * 3 // 2,
                  "output_frames": program.output_frames, "output_regions": program.output_regions,
                  "manifest": str(directory / "program.json"), "manifest_sha256": digest((output / directory / "program.json").read_bytes()),
                  "prediction": tail, "plan": case["plan"], "output_index": case["output_index"],
                  "truth": case["truth"], "all_four_truth_oracle": candidate["truth_oracle"],
                  "scalar_instructions": len(case["plan"]), "scalar_minimum": case["scalar_minimum"],
                  "scalar_minimum_status": "proved function-global minimum attained",
                  "function_global_packed_minimum": case["packed_minimum"],
                  "fixed_construction_depth": candidate["dependency_depth"],
                  "fixed_construction_earliest_schedule": candidate["dependency_layers"],
                  "actual_schedule_pictures": len(program.operations), "costs": costs,
                  "proof_refs": case["proof_refs"], "validation_fixtures": fixture_records,
                  **{key: value for key, value in fixture_records[0].items() if key not in ("label", "truth_pairs_present")}}
        report["cases"].append(record)
        print("PASS %s: 3 instructions, %d pictures, %d prediction pixels, %d slices, %d bytes; %s" %
              (identifier, len(program.operations), costs["prediction_coded_luma_samples"],
               costs["prediction_slices"], costs["prediction_bytes"], ", ".join(ready)), flush=True)
    for mask in (6, 9):
        records = [record for record in report["cases"] if record["mask"] == mask]
        winners = {key: [record["id"] for record in records
                        if record["costs"][key] == min(item["costs"][key] for item in records)]
                   for key in METRICS}
        frontier = pareto_frontier(records)
        report["decisions"].append({"mask": mask, "function": records[0]["function"],
                                    "pareto_candidate_ids": frontier, "metric_winners": winners,
                                    "single_best_candidate": None,
                                    "decision": "Choose the parallel packed candidate for fewer pictures; choose a narrow candidate for fewer coded samples. Byte, slice, and retained-slot costs also differ. No candidate is best for every metric.",
                                    "qualification": SCOPE})
    write_json(output / "joint.json", report)
    # The ordinary portable decoder runner can validate these six primary
    # streams using this alias; purpose=joint excludes throughput timing loops.
    write_json(output / "bundle.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "joint")
    parser.add_argument("--decoders", nargs="+", choices=("ffmpeg", "openh264", "vaapi"),
                        default=["ffmpeg", "openh264", "vaapi"])
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    args = parser.parse_args()
    report = prepare_joint(args.output, tuple(args.decoders), width=args.width, height=args.height)
    print(json.dumps({"status": report["status"], "candidates": len(report["cases"]),
                      "report": str(args.output / "joint.json")}))


if __name__ == "__main__":
    main()
