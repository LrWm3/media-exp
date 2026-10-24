"""Compare shared-IR H.264 spatial fusion against its unfused baseline.

Both variants decode both fixtures against independent task formulas. The
combined portable suite validates every saved picture, including guards and
chroma, using the existing decoder validator. These are static cost savings;
this experiment does not measure latency.
"""

import argparse
import json
from pathlib import Path

from .demo import ROOT, digest, run, write_json
from .export_h264 import export


TASKS = ("dilate-zero", "smooth-zero", "local-range")


def compare(output):
    output = Path(output).resolve()
    reports = {}
    cases = []
    for label, enabled in (("baseline", False), ("fused", True)):
        root = output / label
        reports[label] = run(root, targets=("h264",), tasks=TASKS, h264_fuse_shifts=enabled)
        export(root)
        for case in json.loads((root / "suite.json").read_text())["cases"]:
            case["name"] = label + "/" + case["name"]
            for field in ("stream", "manifest", "expected"):
                case[field] = label + "/" + case[field]
            for info in case["named_outputs"].values():
                info["path"] = label + "/" + info["path"]
            cases.append(case)
    if reports["baseline"]["source_sha256"] != reports["fused"]["source_sha256"]:
        raise AssertionError("Sources changed between comparisons")
    rows = []
    for old, new in zip(reports["baseline"]["cases"], reports["fused"]["cases"]):
        if old["task"] != new["task"] or old["status"] != "pass" or new["status"] != "pass":
            raise AssertionError("Both plans must support the same task")
        for before, after in zip(old["variants"], new["variants"]):
            if {name: plane["sha256"] for name, plane in before["outputs"].items()} != {
                    name: plane["sha256"] for name, plane in after["outputs"].items()}:
                raise AssertionError("Fusion changed a named output")
        for metric in ("computation_pictures", "prediction_bytes"):
            if new["costs"][metric] > old["costs"][metric]:
                raise AssertionError("Fusion regressed " + metric)
        rows.append({"task": old["task"], "baseline": old["costs"], "fused": new["costs"],
                     "fixtures": len(new["variants"]), "named_outputs_equal": True})
    report = {"format": "codec-compute-motion-fusion-1", "status": "pass",
              "timings": "not measured; savings are decoded pictures and prediction bytes",
              "repository_commit": reports["fused"]["repository_commit"],
              "source_sha256": reports["fused"]["source_sha256"],
              "reports": {label: {"path": label + "/validation.json",
                                   "sha256": digest((output / label / "validation.json").read_bytes())}
                          for label in reports},
              "streams": len(cases), "pictures": sum(case["pictures"] for case in cases),
              "decoder_runs": sum(report["decoder_runs"] for report in reports.values()),
              "cases": rows}
    write_json(output / "comparison.json", report)
    write_json(output / "suite.json", {
        "format": "h264-ir-suite-0.1", "selection": "Both fusion plans; all spatial tasks and both fixtures",
        "comparison_sha256": digest((output / "comparison.json").read_bytes()),
        "exporter_sha256": digest(Path(__file__).read_bytes()), "case_count": len(cases), "cases": cases})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "codec_compute/artifacts/motion-fusion")
    print(json.dumps(compare(parser.parse_args().output), indent=2))
