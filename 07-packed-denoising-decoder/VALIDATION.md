# Packed-denoiser validation

The [prepared bundle metadata](assets/bundle.json) records clean implementation
commit `9aba0237f2018951fde9ba9b9060941cb39dc0a5`, source-file hashes, model hash,
input hashes, prediction-tail hashes, stream hashes, geometry, and full-I420
reference hashes. Preparation used the original frozen weights and eight noisy
shape images from `samples(8, seed=200000)`. It explicitly checks decoded seeds
against the supplied pixels and every feature against `FrozenNetwork.predict`.

## Decoder results

Baseline and 2/4/8-tile schedules all passed FFmpeg, OpenH264, VA-API, and NVDEC.
The full NVDEC run checks **944 pictures, 12,813,312 I420 bytes, and 288 named
output planes** across all four schedules. Comparisons include intermediate
luma, every zero separator and unused tile, outer guards, and neutral chroma.
Every timed repetition also compares all returned I420 bytes and named outputs.

The manifest's `output_regions` gives logical rectangles relative to the active
canvas, excluding the outer 16-pixel guard. Named channels 1 and 2 alias the
same computation. Other channels with identical kernels but different thresholds
remain distinct. Full-frame validation establishes correctness for intermediates;
selected readback checks only its supplied pictures.

## Timing scope and samples

[NVDEC results](assets/nvdec-results.json) include all five samples per mode,
one prior warmup per variant/mode, rotated execution order, and actual helper
copy counters. Inputs are concatenated into one session with an IDR reset per
job. There are no concurrent benchmark jobs. The original model and compiler
are absent from the decoding process.

| Schedule | Decoded pictures/batch | Output copies | Output copy bytes | Output-readback median | Full-readback median |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 416 | 64 | 393,216 | 920.67 ms | 933.96 ms |
| 2 tiles | 232 | 56 | 602,112 | 695.14 ms | 617.48 ms |
| 4 tiles | 160 | 40 | 752,640 | 573.74 ms | 594.90 ms |
| 8 tiles | 136 | 24 | 838,656 | 548.60 ms | 564.19 ms |

Eight tiles improve the output-readback median by **1.678×** and full-readback
median by **1.655×**. The selected-output workload produces about **14.6 images/s**,
including startup and verification. These are finite small batches, not isolated
decoder throughput. Selective readback is not consistently faster: the two-tile
measurements show substantial timing variation. Packing reduces elapsed time
here despite a larger canvas. These measurements do not isolate parsing,
decoding, and copying costs inside the decoder helper.

Timed work includes batch assembly/file I/O, process and pipeline startup,
parsing, decoding, actual surface copies, diagnostics, exact returned-I420
comparison, and named-output extraction/comparison. Compilation, input packing,
reference loading, full padding scans, transfer to the decode host, and capture
writing are excluded. Input packing and first-tail construction are measured
separately in bundle metadata. FFmpeg/OpenH264/VA-API copy full pictures before
host selection; only NVDEC uses actual selective surface copies here.

## Automated checks

The new experiment's 11 tests pass using the original PyTorch environment,
without skips. FFmpeg, OpenH264, and VA-API check all 512 binary 3×3 neighborhoods,
zero/one/random/border images, other legal model weights, and tile capacities
1/2/4/8. Every feature also agrees with the original PyTorch checkpoint.
Tests disable host inference during compilation and encoding, reuse cached
prediction tails with changed inputs, validate selected outputs, and reject
corrupted gaps, unused tiles, guards, and chroma. NVDEC validation uses the eight
prepared demo images; the exhaustive neighborhood suite runs on the three local
decoders.

Nine new shared-compiler tests cover named u8 inputs, P/B arithmetic reading
different tiles of the same picture, rectangular images, ±16-pixel reads,
integer reads expressed in quarter-pixel units, long-lived references, identity
programs, invalid selection, and geometry/slice/PCM-size rejection. The local
OpenH264 wrapper rejects a one-picture input-only stream, so that identity case
is checked with FFmpeg and VA-API.

The shared writer's 43 tests pass, including nine new per-block tests and
unchanged legacy stream hashes. All 78 pre-existing compiler tests and 48
existing compiled Life/maze/denoiser/MNIST/motion tests also pass. Neural checkpoint
regressions were rerun with PyTorch enabled. Together these checks cover
**189 unique tests**.

```sh
python3 -m unittest discover -s decoder_core -p 'test_*.py' -q
python3 -m unittest discover -s 05-h264-compiler -p 'test_*.py' -q
03-denoising-decoder/.venv/bin/python -m unittest discover \
  -s 07-packed-denoising-decoder -p 'test_packed.py' -v
```

Capture images use native NVDEC output planes, with 0/1 mapped to 0/255 for
display. Their hashes are checked against the report. Each channel's picture
and tile are shown explicitly; the grid assembles named outputs from their
respective pictures rather than suggesting that all eight finish together.
