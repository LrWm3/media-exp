"""Load unchanged experiment helpers without leaking their generic module names."""

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
sys.path.insert(0, str(REPOSITORY / "05-h264-compiler"))
sys.path.insert(0, str(REPOSITORY))


def load(name):
    qualified = f"_compiled_denoiser_original_{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    dependencies = {"data": load("data")} if name in ("capture", "train") else {}
    previous = {key: sys.modules.get(key) for key in dependencies}
    specification = importlib.util.spec_from_file_location(qualified, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[qualified] = module
    sys.modules.update(dependencies)
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(qualified, None)
        raise
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module
