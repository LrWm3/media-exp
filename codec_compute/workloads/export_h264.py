"""Index saved H.264 tasks for the existing portable NVDEC validator.

Run after demo.py. The index references existing, hash-checked streams, manifests,
named outputs and full I420 files in the same bundle. It does not run inference.
"""

import argparse
import hashlib
import json
from pathlib import Path


def export(root):
    root = Path(root).resolve()
    report_bytes = (root / "validation.json").read_bytes()
    report = json.loads(report_bytes)
    cases = []

    def checked(relative, checksum):
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("Bundle path escapes root")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError("Bundle hash mismatch: " + relative)
        return data

    for task in report["cases"]:
        if task["target"] != "h264" or task["status"] != "pass":
            continue
        for variant in task["variants"]:
            for key in ("stream", "manifest", "expected"):
                checked(variant[key], variant[key+"_sha256"])
            manifest = json.loads(checked(variant["manifest"], variant["manifest_sha256"]))
            height, width = manifest["ir"]["shape"]
            coded_height, coded_width = manifest["coded_shape"]
            outputs = {}
            for name, plane in variant["outputs"].items():
                checked(plane["path"], plane["sha256"])
                outputs[name] = {**plane, "frame": manifest["output_frames"][name],
                                 "x": manifest["guard"], "y": manifest["guard"]}
            cases.append({"name": f"{task['task']}-{variant['index']}",
                          **{key: variant[key] for key in ("stream", "stream_sha256", "manifest", "manifest_sha256",
                                                          "expected", "expected_sha256")},
                          "pictures": manifest["costs"]["total_pictures"], "width": width, "height": height,
                          "coded_width": coded_width, "coded_height": coded_height,
                          "frame_bytes": coded_width*coded_height*3//2, "named_outputs": outputs})
    if not cases:
        raise ValueError("Report contains no validated H.264 streams")
    # The portable validator calls its associated report comparison.json.
    (root / "comparison.json").write_bytes(report_bytes)
    suite = {"format": "h264-ir-suite-0.1", "selection": "Every H.264 workload and both input fixtures",
             "comparison_sha256": hashlib.sha256(report_bytes).hexdigest(),
             "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "case_count": len(cases), "cases": cases}
    (root / "suite.json").write_text(json.dumps(suite, indent=2)+"\n")
    print(f"Indexed {len(cases)} H.264 streams for portable validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    export(parser.parse_args().bundle)
