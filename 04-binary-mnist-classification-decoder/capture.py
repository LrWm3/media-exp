"""Capture actual decoded pooling, hidden bits, and classification decisions."""

from pathlib import Path
import shutil
import subprocess
import tempfile


def render(samples, output, backend):
    width, height = 1024, 540
    output = Path(output)
    backend_label = {"ffmpeg": "FFmpeg software", "openh264": "OpenH264 software", "vaapi": "Intel VA-API hardware"}[backend]
    with tempfile.TemporaryDirectory(prefix="mnist-capture-") as directory:
        directory = Path(directory)
        for frame, (digit, record) in enumerate(samples):
            canvas = bytearray((11, 18, 32) * (width*height))

            def paint(image, left, top, scale, grayscale=False):
                for y, row in enumerate(image):
                    pixels = b"".join(bytes((value,)*3 if grayscale else ((225, 235, 249) if value else (23, 34, 51))) * scale for value in row)
                    for dy in range(scale):
                        start = ((top+y*scale+dy)*width+left)*3
                        canvas[start:start+len(pixels)] = pixels

            original = [digit.pixels[y*28:(y+1)*28] for y in range(28)]
            paint(original, 24, 122, 7, True)
            paint(digit.binary(), 252, 122, 7)
            pool = record["pooled"]
            paint(pool, 480, 122, 196//len(pool))
            paint([record["hidden"]], 24, 402, 32)
            prediction = (3, 8)[record["bit"]]
            correct = prediction == digit.label
            lines = [
                (24, 18, 27, "white", "MNIST / CLASSIFICATION INSIDE H.264"),
                (24, 60, 15, "0xa9bacf", f"{backend_label} | official test image {digit.index} | sample {frame+1}/{len(samples)}"),
                (24, 94, 14, "0xa9bacf", "ORIGINAL MNIST"),
                (252, 94, 14, "0xa9bacf", "BINARY DECODER INPUT"),
                (480, 94, 14, "0x66e8f2", "DECODER POOLED BITS"),
                (728, 122, 16, "0x66e8f2", "DECODED CLASS"),
                (776, 158, 100, "white", str(prediction)),
                (728, 286, 18, "0xa9bacf", f"Decoded bit = {record['bit']}"),
                (728, 318, 18, "0x6ce3ad" if correct else "0xff8a80", f"Label = {digit.label} / {'CORRECT' if correct else 'INCORRECT'}"),
                (24, 368, 15, "0xa9bacf", f"DECODED HIDDEN ACTIVATIONS / {len(record['hidden'])} LEARNED NEURONS"),
                (24, 464, 15, "0xa9bacf", "Frozen binary weights + prediction arithmetic + thresholding -> one decoded class bit"),
                (24, 498, 13, "0xa9bacf", "Pooling, hidden bits and class come from decoder output. Labels and enlargement added afterwards."),
            ]
            filters = ",".join(f"drawtext=font='DejaVu Sans':text='{text}':expansion=none:x={x}:y={y}:fontsize={size}:fontcolor={color}"
                               for x, y, size, color, text in lines)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                            "-i", "pipe:0", "-vf", filters, "-frames:v", "1", "-y", str(directory/f"frame-{frame:03d}.png")],
                           input=canvas, check=True, timeout=60)
        shutil.copyfile(directory/"frame-000.png", output.with_suffix(".png"))
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "1", "-i", str(directory/"frame-%03d.png"),
                        "-filter_complex", "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", "-y", str(output.with_suffix(".gif"))], check=True, timeout=120)
