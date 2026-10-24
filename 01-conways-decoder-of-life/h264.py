"""Compatibility imports for the shared decoder implementation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder_core.bitstream import Bits, Operation, Source, Writer, nal
from decoder_core.decode import available, decode
