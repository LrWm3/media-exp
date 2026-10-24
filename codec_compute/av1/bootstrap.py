"""Build the narrowly patched, pinned libaom serializer outside the source tree."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess

REVISION = "10aece4157eb79315da205f39e19bf6ab3ee30d0"
TAG = "v3.12.1"
SOURCE_URL = "https://aomedia.googlesource.com/aom"
HERE = Path(__file__).resolve().parent


def build_directory() -> Path:
    digest = hashlib.sha256((HERE / "libaom.patch").read_bytes() +
                            (HERE / "bridge.c").read_bytes()).hexdigest()[:16]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "codec-compute" / "av1" / f"{TAG}-{digest}"


def bridge_path() -> Path:
    """Find the bridge; compilation is an explicit setup step, never hidden."""
    path = build_directory() / "bridge"
    if not path.is_file():
        raise RuntimeError(
            "AV1 bridge unavailable. Run: python3 -m codec_compute.av1.bootstrap"
        )
    return path


def bootstrap() -> Path:
    root = build_directory()
    root.mkdir(parents=True, exist_ok=True)
    bridge = root / "bridge"
    if bridge.is_file():
        return bridge
    source, build = root / "aom", root / "build"
    log_path = root / "build.log"
    with log_path.open("ab") as log:
        def run(args: list[str], **kwargs: object) -> None:
            try:
                subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                               check=True, **kwargs)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"AV1 build failed; see {log_path}") from exc

        if not source.exists():
            run(["git", "clone", "--depth", "1", "--branch", TAG, SOURCE_URL,
                 str(source)])
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != REVISION:
            raise RuntimeError(f"Unexpected libaom revision: {revision}")
        # The source belongs solely to this content-addressed build cache.
        run(["git", "-C", str(source), "reset", "--hard", REVISION])
        run(["git", "-C", str(source), "apply", str(HERE / "libaom.patch")])
        run(["cmake", "-S", str(source), "-B", str(build),
             "-DCMAKE_BUILD_TYPE=Release", "-DAOM_TARGET_CPU=generic",
             "-DENABLE_DOCS=OFF", "-DENABLE_TESTS=OFF", "-DENABLE_EXAMPLES=OFF",
             "-DENABLE_TOOLS=OFF", "-DCONFIG_AV1_DECODER=0",
             "-DCONFIG_MULTITHREAD=0", "-DCONFIG_RUNTIME_CPU_DETECT=0",
             "-DCONFIG_INTERNAL_STATS=0", "-DCONFIG_AV1_TEMPORAL_DENOISING=0"])
        run(["cmake", "--build", str(build), "--target", "aom", "-j",
             str(min(os.cpu_count() or 1, 8))])
        temporary = root / "bridge.tmp"
        run([os.environ.get("CC", "cc"), "-O2", "-I", str(source),
             str(HERE / "bridge.c"), str(build / "libaom.a"), "-lm", "-lpthread",
             "-o", str(temporary)])
        temporary.replace(bridge)
    return bridge


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(bootstrap())
