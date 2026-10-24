# Compiled maze validation

Validated on 2026-10-23 with actual FFmpeg, OpenH264, and Intel VA-API decoders.
All 14 integration tests passed; no backend checks were skipped.

```sh
python -m unittest discover -s 02-maze-flood-fill-decoder-solver/compiled -p 'test_*.py' -v
python 02-maze-flood-fill-decoder-solver/compiled/run.py \
  --steps 52 --decoder vaapi --verify --preview \
  --output 02-maze-flood-fill-decoder-solver/compiled/assets/maze.h264
```

The 32×32 demo capture contains 560 decoded pictures and 53 named flood states.
It reaches 359 cells, reaches the goal in 50 hops, and converges at step 51.
The sealed nine-cell room remains unreachable. These distances and every
intermediate state match the original independent BFS implementation.

The 148,540-byte stream was decoded independently by all three backends;
each produced 3,440,640 raw YUV bytes with SHA-256:

```text
dc2efdf335b58faf9ed118e66b9ae0a925d7f1c83cb23fee9a8a955961e3a3a8
```

Every temporary picture retained a zero guard and neutral chroma. The GIF
colors actual Intel VA-API flood states. Its route is reconstructed from
first decoded arrival times; no BFS result supplies capture pixels or routes.

The integration suite additionally checks all 243 local wall/open/reached
configurations for three steps, every local rule on OpenH264 and VA-API,
open boundaries, multiple starts, diagonal isolation, short and zero budgets,
random walls, replacement inputs, disabled-oracle execution, optimized and
unoptimized graphs, rejected negative budgets, and full result parity with
the original experiment's actual decoded stream.

For the same maze and 52-step budget, the original specialized program uses
261 pictures and 78,341 bytes. The compiled variant uses more pictures while
expressing the algorithm solely through shared symbolic APIs.
