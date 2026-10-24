# Life through the shared H.264 compiler

This variant expresses Conway's rule in [the shared compiler DSL](../../05-h264-compiler/README.md).
The [kernel](life_kernel.py) sums eight shifted binary neighbors, tests counts
equal to two and three, and combines those predicates with the current cell.
Shifts use finite zero padding, matching the original experiment's boundaries.

![Life generations captured from the compiled stream on Intel VA-API](assets/life.gif)

The checked-in [stream](assets/life.h264) contains 40 generations of a 32×32
r-pentomino seed. All 41 states match the independent oracle on FFmpeg,
OpenH264, and Intel VA-API; all three decoders produce identical raw bytes.
See [validation results](VALIDATION.md) and the [compiler manifest](assets/life.json).

`h264.compile` traces the symbolic function and lowers it to prediction pictures.
Compilation receives dimensions and a generation budget; seed values enter only
when the program is encoded. Python does not calculate future generations for
the bitstream. The original Life oracle runs only when checking decoder output.

Run from the repository root:

```sh
python 01-conways-decoder-of-life/compiled/run.py \
  --pattern r-pentomino --width 32 --height 32 --generations 40 \
  --decoder ffmpeg --verify --preview \
  --output 01-conways-decoder-of-life/compiled/out/life.h264
```

Use `--decoder openh264` or `--decoder vaapi` for an installed alternative.
`--no-optimize` disables compiler optimizations while preserving the algorithm.
FFmpeg is required to render previews. The CLI saves the `.h264` program and
compiler `.json` manifest, decodes that saved stream, writes `.result.json`, and
optionally creates a `.gif` from the decoded generations.

```python
from life_kernel import compile_life

program = compile_life(32, 32, generations=10)
outputs = program.verify(board, backend="ffmpeg")  # board contains logical 0/1
final = outputs["output"]
first_generation = outputs["generation_1"]
```

Run this snippet from `compiled/`, or add that directory to `sys.path`.
Named outputs include `generation_0` through `generation_N`. Their picture
indices come from `program.output_frames`; compiler-generated schedules have no
fixed pictures-per-generation contract. The compiler owns binary input
normalization, arithmetic lowering, reference allocation, and output extraction.

Integration checks:

```sh
python -m unittest discover \
  -s 01-conways-decoder-of-life/compiled -p 'test_*.py' -v
```

Tests decode every one of the 512 local Life configurations, compare multiple
complete generations with the independent oracle, exercise zero padding and
input reuse, compare the original and compiled streams, and check optimized and
unoptimized graphs. FFmpeg, OpenH264, and VA-API paths are exercised when
available; missing optional backends are reported as skips.

For this capture, the general compiler emits 922 pictures and 240,609 bytes.
The original specialized schedule emits 561 pictures and 153,384 bytes for the
same seed and budget. The compiler version demonstrates a reusable lowering
path; the original hand-tuned schedule remains more compact.
