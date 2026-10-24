# Recorded validation

Environment: Python 3.12, FFmpeg 6.1.1, GStreamer 1.24.2, and Intel graphics through
`/dev/dri/renderD128`. Hardware decoding explicitly selects `vah264dec`; it does
not fall back to software.

## Tests

```sh
python3 -m unittest -v
```

Result: **11 maze tests passed, none skipped** (3.159 seconds). The existing Life
suite also passed all **9 tests**, and regenerating its 120-generation demo
produced byte-identical H.264 after extracting the shared writer.

The maze suite covers:

- All 243 ternary center/cardinal-neighbor configurations on FFmpeg, OpenH264,
  and Intel hardware.
- A full demo flood on all three decoders, including the sealed room and its
  independently verified 50-step shortest route.
- Multiple starts, open picture boundaries, diagonal-only connectivity, random
  walls, zero steps, and an insufficient iteration budget.
- A winding corridor with a 463-hop route, beyond eight-bit distance range.
- Changing the initial maze while preserving byte-identical prediction instructions.
- Compilation and decoded-distance extraction with both BFS oracle entry points
  disabled; no reference solution is needed to generate or solve the stream.
- Invalid maze syntax and sample values.

## Captured demo

```sh
python3 run.py --verify --decoder vaapi --preview
```

| Measurement | Result |
| --- | --- |
| Logical maze | 32×32 cells |
| Coded picture, including guard | 64×64 pixels |
| Fixed iteration budget | 368 |
| Total decoded pictures | 1,841 |
| Verified flood states | 369 |
| Reachable passages | 359 |
| Unreachable passages | 9 |
| Shortest goal distance | 50 hops |
| First unchanged state | Step 51 |
| Executable stream size | 505,889 bytes |
| Annotated capture size | 73,896 bytes |

The full stream was checked against BFS with FFmpeg, OpenH264, and Intel hardware.
Every completed flood state and first-arrival distance matched. Every temporary
picture retained its zero guard and neutral chroma. FFmpeg and OpenH264 also
produced byte-identical complete raw output, including arithmetic intermediates.

The committed capture was created from Intel hardware output, stops at convergence,
then holds the decoded shortest route. Its colors, labels, and route overlay are
presentation added after decoding. The executable stream retains all 368 steps.

Stream SHA-256 (`assets/maze.h264`):

```text
dc82b3b0364aef29b324463495b7a818768706420ec76513f805bf9eb32e3374
```

FFmpeg/OpenH264 complete raw I420 SHA-256:

```text
9cb44361f97ed059212ee8a5544d4d2ddd30cb2d3fd97a03f728d56bd96ebeac
```

The shared writer retains the profile, arithmetic, and bitstream limits reviewed
in [experiment 01](../01-conways-decoder-of-life/VALIDATION.md). This experiment
adds no new H.264 syntax; its new weights and byte-valued seed are within those
bounds. No official conformance suite or other hardware families were tested.
