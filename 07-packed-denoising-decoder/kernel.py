"""Pack the existing frozen denoiser's lowered operations into shared pictures."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03-denoising-decoder" / "compiled"))

from denoiser import MODEL_PATH, FrozenNetwork, compile_model as compile_baseline
from denoiser_original import load
import h264


def compile_denoiser(model=None, *, width=32, height=32, tiles=8):
    """Reuse frozen weights and ordinary lowering, then pack independent work."""
    return h264.pack(compile_baseline(model, width=width, height=height), tiles=tiles)
