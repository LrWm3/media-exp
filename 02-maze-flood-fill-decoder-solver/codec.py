"""Make the repository's shared codec available when running from this folder."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decoder_core.bitstream import Operation, Source, Writer
from decoder_core.decode import available, decode
