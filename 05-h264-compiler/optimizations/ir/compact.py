#!/usr/bin/env python3
"""Export the first baseline/optimized pair per graph for portable validation.

The full comparison report remains unchanged. The new suite lists only the
copied streams, and records its original suite hash and representative scope.
Both validate.py and ../benchmark.py consume the resulting bundle unchanged.
"""

import argparse
import json
from pathlib import Path

from validate import checked, digest


def compact(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if bundle == output:
        raise ValueError("Compact destination must differ from source bundle")
    suite_bytes = (bundle / "suite.json").read_bytes()
    original = json.loads(suite_bytes)
    report_bytes = checked(bundle, "comparison.json", original["comparison_sha256"])
    cases = [group[variant] for group in original["representatives"].values()
             for variant in ("baseline", "optimized")]
    output.mkdir(parents=True, exist_ok=True)
    copied = {}

    def copy(relative, checksum):
        if relative in copied:
            if copied[relative] != checksum:
                raise AssertionError("Conflicting source hashes: " + relative)
            return
        data = checked(bundle, relative, checksum)
        path = (output / relative).resolve()
        if output not in path.parents:
            raise ValueError("Destination path escapes compact bundle")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        copied[relative] = checksum

    for case in cases:
        for key in ("stream", "manifest", "expected"):
            copy(case[key], case[key + "_sha256"])
        for info in case["named_outputs"].values():
            copy(info["path"], info["sha256"])
    (output / "comparison.json").write_bytes(report_bytes)
    suite = {**original, "case_count": len(cases), "cases": cases,
             "selection": "First segment baseline/optimized pair per graph; full comparison report retained",
             "original_suite_sha256": digest(suite_bytes),
             "original_case_count": original["case_count"],
             "compactor_sha256": digest(Path(__file__).read_bytes())}
    (output / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    print(f"Copied {len(cases)} representative streams; original suite has {original['case_count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact(args.bundle, args.output)
