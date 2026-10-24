# Five pictures per expansion

Let `C(x) = min(255, max(0, x))`. State `s` is 0 for a wall, 128 for an unvisited
passage, and 255 for a reached cell. All weight denominators are zero.

| Picture | Operation | Weights | Offsets |
| --- | --- | --- | --- |
| 1 | `T = P(s)` | 1 | −128 |
| 2 | `NS = B(T north, T south)` | 5, 5 | 0, 0 |
| 3 | `EW = B(T east, T west)` | 5, 5 | 0, 0 |
| 4 | `R = B(NS, EW)` | 2, 2 | 0, 0 |
| 5 | `next = B(R, s)` | 1, 4 | −128, −128 |

P prediction computes `C(w*x + offset)`. B prediction computes
`C(((w0*x + w1*y + 1) >> 1) + ((o0 + o1 + 1) >> 1))`.
These are the same prediction primitives used in [experiment 01](../01-conways-decoder-of-life/DESIGN.md).

`T` maps states `0,128,255` to `0,0,127`. In a neighbor pair, any nonzero value
produces at least `((5*127 + 1) >> 1) = 318`, clipped to 255. Therefore `NS` and
`EW` are exact Boolean OR results represented as 0/255. Adding them with
saturation produces `R`, which is 255 if any cardinal neighbor is reached.

The final picture computes:

```text
next = C(((R + 4*s + 1) >> 1) - 128)
```

| Current cell | No reached neighbor (`R=0`) | Reached neighbor (`R=255`) |
| --- | --- | --- |
| Wall, 0 | 0 | 0 |
| Unvisited, 128 | 128 | 255 |
| Reached, 255 | 255 | 255 |

The center and four neighbors each have three states: the integration tests cover
all `3^5 = 243` local configurations using real decoders. Negative offsets and
positive weights fit H.264's ranges; B weight sums are 10, 4, and 5.

## Data and instruction separation

The I_PCM seed contains both the maze and the starts. No later picture carries
new pixel data or residual coefficients. For equal dimensions and step count,
all subsequent instruction bytes are identical across mazes. A seed-swap test
checks this and decodes the changed maze correctly.

The shared [writer](../decoder_core/bitstream.py) uses High profile level 3.1,
integer-pixel motion vectors, disabled deblocking, and explicit reference lists.
Its 16-pixel zero guard makes the outside impassable. All operands are at most
five pictures old, fitting the 16-picture sliding reference buffer. Initial
sample values now support all bytes; experiment 01's binary seed output remains
byte-identical to its committed stream.

Each completed iteration extends reachability by exactly one edge. By induction,
after iteration `k`, reached cells are exactly those within `k` hops of a source.
Walls never change, reached cells persist, and diagonal movement is impossible.

## Distances, routes, and termination

The decoder computes reachability, not a numeric distance image. `solve.py`
records each cell's first reached iteration as a Python integer, so distances
can exceed 255. The 463-hop corridor test covers this case. A shortest path is
read back by following cardinal neighbors with distance one less, ending at one
of the starts. This postprocessing does not run BFS or alter decoded states.

The bitstream has a fixed iteration count; it cannot halt itself on convergence.
With `P` open cells, `P` steps guarantee all reachable cells have been found and
at least one unchanged state has been observed. A smaller explicit `--steps`
budget is supported. Until two consecutive states are equal, null distances mean
unresolved; afterwards they mean unreachable. JSON uses −1 for walls.

The GIF selects every fifth decoded picture, stops at the first unchanged state,
and holds the result with its extracted path. Colors, text, and the route overlay
are added afterwards. They are not computation performed by H.264. The raw stream
contains all instruction pictures, including any unused iteration budget.

## Limits

The CLI uses 30 pictures/second, corresponding to six expansions/second when
played at its original timing. Offline decoding may run faster. The one-slice-per-
prediction-macroblock construction limits the padded picture to 60 macroblocks
at that rate. A 32×32 maze fits; 64×64 fits; 96×96 does not. The writer also checks
reference-buffer, per-axis, and initial PCM access-unit bounds. There is no wrap
at boundaries, no diagonal movement, and no weighted edges.
