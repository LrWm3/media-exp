#!/usr/bin/env python3
"""Explicit native build with compiler, flags, vectorization and binary provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def build(output, compiler="c++"):
    root = Path(__file__).resolve().parents[3]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([compiler, "--version"], universal_newlines=True).splitlines()[0]
    report = {"compiler": version, "flags": {}, "source_sha256": {}, "helper_sha256": {}, "diagnostics": {}}
    for name, source in (("cpu", "codec_compute/workloads/cpu_comparison/cpu.cpp"),
                         ("cpu_planes", "codec_compute/workloads/cpu_comparison/cpu_planes.cpp"),
                         ("pcm", "decoder_core/pcm_native.cpp")):
        flags = ["-std=c++17", "-O3", "-DNDEBUG", "-fPIC", "-shared", "-Wall", "-Wextra"]
        if name != "pcm" and "clang" not in version.lower():
            flags += ["-fopt-info-vec-optimized"]
        target = output/("lib"+name+".so")
        completed = subprocess.run([compiler, *flags, source, "-o", str(target)], cwd=root,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=True)
        report["flags"][name] = flags
        report["source_sha256"][source] = hashlib.sha256((root/source).read_bytes()).hexdigest()
        report["helper_sha256"][name] = hashlib.sha256(target.read_bytes()).hexdigest()
        report["diagnostics"][name] = completed.stdout+completed.stderr
    (output/"build.json").write_text(json.dumps(report, indent=2)+"\n")
    print(output/"build.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()
    build(args.output, args.compiler)
