"""Color actual decoded states; overlay a path derived from first-arrival times."""

from pathlib import Path
import subprocess


def render(solution, maze, output, backend):
    palette = {0: (14, 22, 36), 128: (61, 75, 94), 255: (35, 106, 165)}
    starts = set(maze.starts)
    end = solution.converged_at if solution.converged_at is not None else len(solution.states) - 1
    states = solution.states[:end + 1]
    route = set(solution.path)
    frames = []
    for step in range(len(states) + 24):
        index = min(step, len(states) - 1)
        state = states[index]
        previous = states[index - 1] if index else bytes(len(state))
        rgb = bytearray()
        for position, value in enumerate(state):
            point = (position % maze.width, position // maze.width)
            color = palette[value]
            if value == 255 and previous[position] != 255:
                color = (102, 232, 242)
            if step >= len(states) and point in route:
                color = (255, 209, 92)
            if point in starts:
                color = (79, 222, 145)
            elif point == maze.goal:
                color = (246, 115, 120)
            rgb.extend(color)
        frames.append(rgb)
    scale = max(1, 512 // max(maze.width, maze.height))
    grid_w, grid_h = maze.width * scale, maze.height * scale
    canvas_w = max(560, grid_w + 48)
    canvas_h = grid_h + 176
    backend_label = {"vaapi": "Intel VA-API hardware", "ffmpeg": "FFmpeg software", "openh264": "OpenH264 software"}[backend]
    goal_distance = len(solution.path) - 1 if solution.path else None
    if goal_distance is not None:
        result_label = f"Goal distance = {goal_distance} steps"
    else:
        result_label = "Goal unreachable" if solution.converged_at is not None else "Goal not reached within budget"
    if maze.goal is None:
        result_label = "Multiple-source reachability" if len(maze.starts) > 1 else "Reachability"
    # Text only annotates the capture; the solver never reads these overlays.
    filters = (
        f"scale={grid_w}:{grid_h}:flags=neighbor,pad={canvas_w}:{canvas_h}:24:104:color=0x0b1220,"
        "drawtext=font='DejaVu Sans':text='MAZE / H.264 DECODER':x=24:y=20:fontsize=24:fontcolor=white,"
        f"drawtext=font='DejaVu Sans':text='{backend_label} | {maze.width} x {maze.height} cells':x=24:y=56:fontsize=14:fontcolor=0xa9bacf,"
        f"drawtext=font='DejaVu Sans':text='Wave step %{{eif\\:min(n\\,{end})\\:d}} / {end}':x=24:y=79:fontsize=14:fontcolor=0x66e8f2,"
        f"drawtext=font='DejaVu Sans':text='{result_label}':x=24:y={grid_h + 120}:fontsize=16:fontcolor=0xffd15c,"
        f"drawtext=font='DejaVu Sans':text='Blue = reached   Cyan = frontier   Gold = shortest path':x=24:y={grid_h + 146}:fontsize=12:fontcolor=0xa9bacf"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{maze.width}x{maze.height}", "-framerate", "12", "-i", "pipe:0",
                    "-filter_complex", f"[0:v]{filters},split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none",
                    "-loop", "0", "-y", str(output)], input=b"".join(frames), check=True, timeout=120)
