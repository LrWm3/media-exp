# Maze flood through the shared H.264 compiler

This variant expresses flood propagation in [the shared compiler DSL](../../05-h264-compiler/README.md).
The [kernel](maze_kernel.py) derives an open-cell mask and a start mask from the
input, ORs reachability with four shifted neighbors, then ANDs with open cells.
Finite zero padding prevents movement across opposite boundaries.

![Flood states and route captured from the compiled stream on Intel VA-API](assets/maze.gif)

The checked-in [stream](assets/maze.h264) reaches the goal in 50 hops and
converges at step 51, leaving the sealed nine-cell room unreachable. FFmpeg,
OpenH264, and Intel VA-API produce identical raw bytes; every flood state and
arrival distance matches independent BFS. See [validation results](VALIDATION.md),
the [compiler manifest](assets/maze.json), and [decoded distances](assets/maze.distances.json).

The input keeps the original byte encoding: walls are `0`, open cells are
`128`, and starts are `255`. Symbolic thresholds produce logical `0/1` masks.
The graph depends on dimensions and expansion budget; its instruction sequence
does not depend on maze contents. The H.264 decoder computes every flood state.

Run from the repository root:

```sh
python 02-maze-flood-fill-decoder-solver/compiled/run.py \
  --steps 52 --decoder ffmpeg --verify --preview \
  --output 02-maze-flood-fill-decoder-solver/compiled/out/maze.h264
```

`--maze PATH` loads an ASCII maze; `--random-seed N` chooses another fixture.
With no `--steps`, the open-cell count supplies a convergence-safe budget.
Use `--decoder openh264` or `--decoder vaapi` for an installed alternative.
`--no-optimize` disables compiler optimizations. FFmpeg renders previews.

The CLI saves the `.h264` stream and compiler `.json` manifest, decodes the saved
stream, and writes `.distances.json`. An optional `.gif` colors decoded states
and overlays the shortest route. Unvisited cells remain unresolved when the
budget ends before convergence.

```python
from maze_kernel import compile_maze, extract_solution

program = compile_maze(maze.width, maze.height, steps=52)
outputs = program.verify(maze.initial_luma(), backend="ffmpeg")
solution = extract_solution(outputs, maze, steps=52)
```

Run this snippet from `compiled/`, or add that directory to `sys.path`.
`step_0` through `step_N` are decoded binary reachability grids; `passages`
is the decoded open-cell mask. Output picture indices are recorded in
`program.output_frames`. Extraction records the first decoded step that reaches
each cell, then walks decreasing distances to recover a route. It performs no
BFS. The independent BFS implementation runs only for verification.

Integration checks:

```sh
python -m unittest discover \
  -s 02-maze-flood-fill-decoder-solver/compiled -p 'test_*.py' -v
```

Tests cover all 243 local wall/open/reached configurations, every intermediate
state against BFS, sealed rooms, multiple starts, open boundaries, short and
zero budgets, input reuse, original-stream parity, and disabled-oracle execution.
FFmpeg, OpenH264, and VA-API paths are exercised when available; missing optional
backends are reported as skips.

For this 52-step capture, the general compiler emits 560 pictures and 148,540
bytes. The original specialized schedule emits 261 pictures and 78,341 bytes
for the same maze and budget. Both compute the same states; the shared
compiler trades the specialized schedule's compactness for reusable operations.
