# Conway's decoder of Life: technical design

**Working experiment:** an ordinary H.264 decoder calculates Conway's Game of
Life from an initial grid. Python writes the instructions; the decoder computes
the future states. The generated stream has no residual coefficients or new
pixel samples after its initial I_PCM picture.

Verified with FFmpeg software decoding, OpenH264, and Intel VA-API hardware
decoding on this machine. Each Life generation takes **14 decoded pictures**.
This demonstrates the construction on these decoders, not universal hardware
compatibility or certification by a conformance test suite.

## Run

Requires Python 3.10+ and FFmpeg for verification and previews. The generator
itself uses only Python's standard library. Run commands from this directory:

```sh
python3 run.py --verify --preview
python3 -m unittest -v
```

The default run creates:

- `out/life.h264`: executable video bitstream, including intermediate pictures.
- `out/life.json`: initial grid and readable instruction manifest.
- `out/life.gif`: 120 generations of an R-pentomino, selected from actual decoded
  pixels, cropped and enlarged. The GIF is a preview, not the executable program.

Other starting patterns:

```sh
python3 run.py --pattern glider --generations 80 --verify --preview
python3 run.py --pattern random --random-seed 42 --verify
```

Width and height are multiples of 16, defaulting to 64 each. A cell occupies one
luma pixel. The board has finite, permanently dead boundaries; it does not wrap.
At the default 30 pictures per second, the padded picture may contain at most
60 macroblocks because each prediction macroblock uses a separate slice. For
example, 64×64 and 128×64 boards fit; 96×96 does not. The writer also checks the
level's dimensions, reference-buffer capacity, and initial PCM access-unit size.
For an arbitrary initial grid, call `program.compile_stream(board, generations)`
with a rectangular list of lists containing `0` and `1`.

Optional independent decoder checks require GStreamer and the relevant plugins:

```sh
python3 run.py --verify --decoder vaapi
python3 run.py --verify --decoder openh264
```

The hardware path explicitly uses `vah264dec`, so it cannot silently fall back to
a software decoder. Test cases skip optional backends when their decoder elements
are unavailable. A listed backend that fails to decode fails the test.

To watch the executable stream while showing only generation pictures:

```sh
ffplay -vf "select='not(mod(n,14))',crop=iw-32:ih-32:16:16,scale=512:-1:flags=neighbor,setpts=N/(12*TB)" out/life.h264
```

Opening the raw stream without that filter also shows the temporary calculations.
Its timing is 30 pictures per second, or approximately 2.14 Life generations per
second. Every intermediate picture still gets decoded when the display filter
selects only generation pictures.

## Exact arithmetic

Write `C(x) = min(255, max(0, x))`. Weighted P prediction with denominator zero
implements `C(w*x + offset)`. Explicit weighted B prediction with denominator zero
and zero offsets implements `C((w0*x + w1*y + 1) >> 1)`. Choosing even weights
makes the latter exact integer arithmetic, with no rounding error.

Let `c` be the current cell, in `{0,1}`, and `n` its living-neighbor count, in
`{0,...,8}`. Life's rule is precisely:

```text
next cell lives iff 5 <= 2*n + c <= 7
```

The following fixed program runs identically for every pixel and every seed:

| Picture within generation | Calculation | H.264 operation |
| --- | --- | --- |
| 1 | `c = C(((old + 64) >> 7) - 1)` | P weight 1, denominator 7, offset -1; converts 0/255 to 0/1 |
| 2–8 | Sum eight shifted versions of `c` | Seven B pictures with weights 2,2; integer-pixel motion vectors |
| 9 | `s = 2*n + c` | B weights 4,2 |
| 10 | `a = C(s - 4)` | P weight 1, offset -4 |
| 11 | `b = C(s - 7)` | P weight 1, offset -7 |
| 12 | `t = C(a - 4*b)` | B weights 2,-8 |
| 13 | `u = C(127*t)` | P weight 127 |
| 14 | `next = C(3*u)` | P weight 3; returns exactly 0 or 255 |

For `s <= 4`, `t = 0`. For `s = 5,6,7`, `t = 1,2,3`. For `s >= 8`,
`a - 4*b = 24 - 3*s <= 0`, so `t = 0`. The two final multiplications saturate
every positive `t` to 255. This covers all 18 combinations of center state and
neighbor count, including a live cell with two neighbors.

## Bitstream details

- High profile, level 3.1, 8-bit 4:2:0, CAVLC, progressive frames.
- The initial picture uses I_PCM, storing only the initial condition. High profile
  matters: Main profile forbids zero-valued I_PCM samples.
- Every subsequent macroblock uses `P_L0_16x16` or `B_Bi_16x16`, with
  `coded_block_pattern = 0`. There are no residuals, transforms, or future grids.
- One slice per prediction macroblock makes motion-vector predictors zero and allows the
  guard macroblocks to use zero weights independently of the interior.
- The seed uses a single slice containing all its I_PCM macroblocks, satisfying
  the tighter slice-count limit on the initial picture.
- Luma motion vectors move by whole pixels, avoiding interpolation. Chroma stays
  neutral at 128 and carries no computation.
- A 16-pixel guard surrounds the board. Zero-weight prediction resets it in every
  picture, preventing reference-edge replication from changing the boundary rule.
- Deblocking is disabled. Full-range luma preserves 0 and 255.
- Both B reference lists are explicitly set; both may reference earlier pictures.
  Every picture is a reference picture. A sliding buffer of 16 pictures is
  declared; operands are never more than eight pictures old.
- Picture order equals decoding order. SPS timing and buffer restrictions are
  signaled. Frame numbers and picture-order LSBs use 16-bit fields.

The generator unrolls a finite instruction sequence. There is no decoder-side
branch, dynamic loop, feedback from Python, or conventional Life computation
inside the generator. `life.step()` is called only after decoding, for validation.

## Evidence

The integration suite checks every pixel in each generation, plus the zero guard
and neutral chroma in **every** intermediate picture. It includes:

- All 512 distinct 3×3 binary neighborhoods, decoded by FFmpeg, OpenH264, and
  Intel VA-API.
- Empty grid, still life, oscillator, glider, R-pentomino, and random boards,
  including cells at boundaries, for 25 generations with FFmpeg.
- Random board for 25 generations on Intel hardware.
- A changed initial picture followed by **byte-identical prediction instructions**:
  the decoder computes the changed future correctly for 12 generations.
- Zero-generation seed decoding and reference-distance checks for a 5,000-generation
  instruction sequence. That last check is structural; it does not decode the long
  sequence or establish frame-counter wrap behavior.
- Rejection of inputs that violate per-axis dimensions, slice rate, or the
  initial access-unit size bound for High profile level 3.1.

The checked demo, `out/r-pentomino.h264`, contains 1,681 decoded pictures for
120 transitions. All 121 states match the independent Python oracle. Scratch
outputs under `out/` are ignored by Git; the checked stream and a capture from
hardware decoding are committed under `assets/`. Recreate the stream with:

```sh
python3 run.py --generations 120 --verify --preview --output out/r-pentomino.h264
```

Validated locally with FFmpeg 6.1.1 and GStreamer 1.24.2, including Intel graphics
through `/dev/dri/renderD128`. See `VALIDATION.md` for the recorded run.

## Sources

- [ITU-T H.264 recommendation](https://www.itu.int/rec/T-REC-H.264), specifically
  the [August 2024 edition](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.264-202408-S!!PDF-E&lang=e&type=items):
  weighted prediction (§§7.4.3.2, 8.4.2.3.2, 8.4.3), reference lists and marking
  (§§8.2.4, 8.2.5), profiles and levels (Annex A), and VUI restrictions (§E.2.1).
- [FFmpeg H.264 slice decoder](https://github.com/FFmpeg/FFmpeg/blob/n6.1/libavcodec/h264_slice.c)
  and [CAVLC decoder](https://github.com/FFmpeg/FFmpeg/blob/n6.1/libavcodec/h264_cavlc.c).
- [GStreamer VA-API H.264 decoder](https://gstreamer.freedesktop.org/documentation/va/vah264dec.html).
