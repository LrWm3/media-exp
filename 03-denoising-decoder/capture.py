"""Present clean/noisy/decoder images and actual decoded feature maps."""

from pathlib import Path
import shutil
import subprocess
import tempfile

from data import error_rate


def render(clean, noisy, decoded, features, output, backend):
    width, height = 864, 568
    background, foreground = (11, 18, 32), (225, 235, 249)
    backend_label = {"vaapi": "VA-API hardware", "ffmpeg": "FFmpeg software", "openh264": "OpenH264 software"}[backend]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="denoising-capture-") as directory:
        directory = Path(directory)
        for index, (target, source, result, maps) in enumerate(zip(clean, noisy, decoded, features)):
            canvas = bytearray(background * (width * height))

            def paint(image, left, top, scale):
                for y, row in enumerate(image):
                    expanded = b"".join(bytes(foreground if value else (19, 29, 45)) * scale for value in row)
                    for dy in range(scale):
                        begin = ((top + y * scale + dy) * width + left) * 3
                        canvas[begin:begin + len(expanded)] = expanded

            scale = max(1, min(256 // len(source), 256 // len(source[0])))
            for image, left in ((target, 24), (source, 304), (result, 584)):
                paint(image, left, 120, scale)
            for channel, feature in enumerate(maps[:8]):
                paint(feature, 24 + channel * 104, 464, max(1, 64 // len(feature)))
            noisy_error, output_error = error_rate([source], [target]), error_rate([result], [target])
            lines = [
                (24, 18, 26, "white", "DENOISING / H.264 DECODER"),
                (24, 58, 14, "0xa9bacf", f"PyTorch-trained binary CNN | {backend_label} | sample {index + 1} / {len(clean)}"),
                (24, 92, 15, "0xa9bacf", "CLEAN TARGET"),
                (304, 92, 15, "0xa9bacf", "NOISY INPUT"),
                (584, 92, 15, "0x66e8f2", "DECODER OUTPUT"),
                (24, 394, 18, "0xffd15c", f"Pixel errors = {noisy_error:.2%} input -> {output_error:.2%} decoded"),
                (24, 434, 14, "0xa9bacf", f"DECODED 3x3 FEATURES / {min(8, len(maps))} OF {len(maps)} CHANNELS"),
            ]
            for channel in range(min(8, len(maps))):
                lines.append((24 + channel * 104, 540, 12, "0xa9bacf", f"CH {channel + 1}"))
            filters = ",".join(f"drawtext=font='DejaVu Sans':text='{text}':expansion=none:x={x}:y={y}:fontsize={size}:fontcolor={color}"
                               for x, y, size, color, text in lines)
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{width}x{height}", "-i", "pipe:0", "-vf", filters,
                            "-frames:v", "1", "-y", str(directory / f"frame-{index:03d}.png")],
                           input=bytes(canvas), check=True, timeout=60)
        shutil.copyfile(directory / "frame-000.png", output.with_suffix(".png"))
        subprocess.run(["ffmpeg", "-v", "error", "-framerate", "1/2", "-i", str(directory / "frame-%03d.png"),
                        "-filter_complex", "[0:v]split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", "-y", str(output.with_suffix(".gif"))], check=True, timeout=120)
