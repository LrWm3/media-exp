# Conway's decoder of Life

An H.264 decoder computes Conway's Game of Life. Only the initial grid is stored
as pixels; subsequent pictures contain prediction instructions, with no residual
pixel data. Each generation takes **14 decoded pictures**.

![Game of Life computed by an H.264 hardware decoder](assets/decoder-capture.gif)

*Capture: 120 generations from Intel VA-API hardware decoding. Every 14th picture
is shown, enlarged; intermediate calculations are omitted. All 121 states match
an independent Life implementation.*

## How it works

1. **Move:** motion vectors read the eight neighboring pixels from a reference picture.
2. **Count:** weighted prediction adds their values to obtain the live-neighbor count.
3. **Decide:** weighted arithmetic and pixel clipping implement the Life rule exactly.
   For neighbor count `n` and cell state `c`, the next cell lives when `5 ≤ 2n+c ≤ 7`.

Reference pictures hold intermediate results. Python generates a fixed sequence
of instructions; the decoder calculates every future grid. The board has dead
boundaries, and the number of generations is fixed when generating the file.

## Try it

Requires Python 3.10+ and FFmpeg. From this folder:

```sh
python3 run.py --verify --preview
python3 -m unittest -v
```

Outputs appear in `out/`. Optional hardware verification uses
`--decoder vaapi` with GStreamer and its VA-API plugin.

[Download the executable H.264 stream](assets/conways-life.h264), or play it
while displaying only completed generations:

```sh
ffplay -vf "select='not(mod(n,14))',crop=iw-32:ih-32:16:16,scale=512:-1:flags=neighbor,setpts=N/(12*TB)" assets/conways-life.h264
```

Verified with FFmpeg, OpenH264, and Intel hardware, including all 512 possible
neighborhoods. [Detailed construction](DESIGN.md) · [Validation results](VALIDATION.md).
