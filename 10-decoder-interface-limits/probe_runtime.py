#!/usr/bin/env python3
"""Record interface availability without changing settings or identifying hardware."""
import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def probe():
    names = ("cuvidCreateDecoder", "cuvidDecodePicture", "cuvidRegisterDecodeSurfaces", "cuvidDecodePictureAsync")
    library = None
    candidate = ctypes.util.find_library("nvcuvid")
    for name in dict.fromkeys(n for n in (candidate, "libnvcuvid.so.1", "libnvcuvid.so") if n):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            pass
    properties = ("enable-full-frame", "disable-dpb", "num-extra-surfaces", "capture-io-mode")
    inspector = shutil.which("gst-inspect-1.0")
    plugin = None
    if inspector:
        plugin = subprocess.run([inspector, "nvv4l2decoder"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True, timeout=30)
    available = plugin is not None and plugin.returncode == 0
    return {
        "format": "codec-compute-interface-availability-1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "nvdecode_library_loadable": library is not None,
        "nvdecode_symbols": {name: library is not None and hasattr(library, name) for name in names},
        "stateful_nvdec_plugin_available": available,
        "stateful_properties": {name: bool(available and re.search(r"^\s*"+re.escape(name)+r"\s*:", plugin.stdout, re.M)) for name in properties},
        "scope": "Library symbol and plugin-property inspection only; no decoder capability query, surface import, CUDA consumer, or API feature execution. No settings modified; no device identity recorded.",
        "interpretation": "Missing interfaces limit this installed software path; they do not prove hardware impossibility. Present symbols do not prove a device supports the feature."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
