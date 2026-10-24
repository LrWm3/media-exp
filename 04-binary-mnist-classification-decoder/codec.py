"""Shared H.264 writer and ordinary decoders."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
