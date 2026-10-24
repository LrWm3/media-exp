"""Inputs, pixel captures, and provenance shared by native-operation examples."""

import hashlib
import math
from pathlib import Path
import platform
import subprocess

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
BACKENDS = ("ffmpeg", "vaapi", "openh264")


def plane_bytes(board):
    return bytes(value for row in board for value in row)


def digest(board):
    return hashlib.sha256(plane_bytes(board)).hexdigest()


def save_plane(board, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (len(board[0]), len(board)), plane_bytes(board)).save(path)


def fixture(width=64, height=64):
    """An image-like grayscale scene with gradients, hard edges, and texture."""
    a, b = [], []
    for y in range(height):
        ar, br = [], []
        for x in range(width):
            value = 20 + 150*x/max(1, width-1) + 35*y/max(1, height-1)
            if (x-width*.28)**2+(y-height*.34)**2 < (width*.16)**2:
                value = 220
            if width*.53 < x < width*.82 and height*.42 < y < height*.80:
                value = 45 if ((x+y)//4) % 2 else 110
            ar.append(round(value))
            br.append(min(255, round(16+115*(.5+.5*math.sin(x*.15+y*.09))
                                     +70*(abs(x-width*.65) < width*.08))))
        a.append(ar)
        b.append(br)
    return {"a": a, "b": b}


def tile_fixture(width=64, height=64):
    cols, rows = width//16, height//16
    result = []
    for y in range(height):
        row = []
        for x in range(width):
            tile = (y//16)*cols+x//16
            v = 25+round(190*tile/max(1, cols*rows-1))
            if x % 16 in (0, 15) or y % 16 in (0, 15):
                v = 255
            elif x % 16 == y % 16 or (x % 16 < 5 and y % 16 < 5):
                v = 0
            row.append(v)
        result.append(row)
    return result


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(title, subtitle, panels, path):
    """Nearest-neighbor enlargement; displayed grayscale values stay unchanged."""
    cols = min(4, len(panels))
    rows = math.ceil(len(panels)/cols)
    panel_w, image_h = 288, 256
    canvas = Image.new("RGB", (32+cols*panel_w, 110+rows*318), "#111b2b")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), title, fill="#edf4ff", font=font(23))
    draw.text((20, 53), subtitle, fill="#a7bed7", font=font(13))
    for index, (label, board) in enumerate(panels):
        left, top = 20+(index % cols)*panel_w, 100+(index//cols)*318
        draw.text((left, top), label, fill="#d9e8fa", font=font(14))
        plane = Image.frombytes("L", (len(board[0]), len(board)), plane_bytes(board))
        target_width = min(256, round(image_h*len(board[0])/len(board)))
        target_height = round(target_width*len(board)/len(board[0]))
        plane = plane.resize((target_width, target_height), Image.Resampling.NEAREST)
        canvas.paste(plane.convert("RGB"), (left, top+28))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def provenance():
    def output(command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            return result.stdout.strip()
        except FileNotFoundError:
            return "unavailable"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "ffmpeg": output(["ffmpeg", "-version"]).splitlines()[0],
            "gstreamer": output(["gst-launch-1.0", "--version"]),
            "display_devices": [line for line in output(["lspci", "-nn"]).splitlines()
                                if any(kind in line for kind in ("VGA", "Display", "3D controller"))],
            "hardware_backend": "Intel VA-API (GStreamer vah264dec)",
            "nvidia_tested": False}
