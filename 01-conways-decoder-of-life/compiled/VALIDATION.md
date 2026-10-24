# Compiled Life validation

Validated on 2026-10-23 with actual FFmpeg, OpenH264, and Intel VA-API decoders.
All nine integration tests passed; no backend checks were skipped.

```sh
python -m unittest discover -s 01-conways-decoder-of-life/compiled -p 'test_*.py' -v
python 01-conways-decoder-of-life/compiled/run.py \
  --pattern r-pentomino --width 32 --height 32 --generations 40 \
  --decoder vaapi --verify --preview \
  --output 01-conways-decoder-of-life/compiled/assets/life.h264
```

The capture contains 922 decoded pictures and 41 named generation states.
The 240,609-byte stream was decoded independently by all three backends;
each produced 5,664,768 raw YUV bytes with SHA-256:

```text
debb74f20aab41a4021433ec9ea110c56836b255f1dfb66ed62b9b81f448bf05
```

Every named state matched the original Python Life oracle. Every temporary
picture retained a zero guard and neutral chroma. The GIF uses actual Intel
VA-API generation pixels, scaled with nearest-neighbor sampling.

The integration suite additionally checks all 512 local configurations for
three complete generations, every local rule on OpenH264 and VA-API, known
patterns and random boundary cells, replacement seeds, zero generations,
optimized/unoptimized equivalence, rejected negative budgets, and pixel parity
with the original experiment's actual decoded stream.

For the same 32×32 seed and 40-generation budget, the original specialized
program uses 561 pictures and 153,384 bytes. The compiled variant uses more
pictures while expressing the algorithm solely through shared symbolic APIs.
