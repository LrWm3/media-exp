# Validation and measurements

## Correctness

The deterministic 24-pair fixture has 25 grayscale inputs, 40 pictures per job,
960 pictures total, 120 named outputs, and 17,694,720 bytes of packed I420.
Every full picture, including guard and actual chroma bytes, agrees on FFmpeg,
OpenH264, Intel VA-API, and NVDEC. Each named result also
agrees with an independent direct-pixel oracle and the compiler graph interpreter.
The prediction tail is identical for every job; inputs alone change.

**65 local tests passed without skips**: nine motion tests, 16 shared decoder
tests, ten shared writer tests, and 30 compiler tests. The new motion suite
performs 99 actual decoder runs across FFmpeg, OpenH264, and VA-API:

- Exhaustive absolute differences and thresholding for all 65,536 byte pairs.
- Cutoff boundaries in both subtraction directions at 1, 32, 128, and 255.
- All 512 local binary neighborhoods for 3 × 3 erosion/dilation, plus zero borders.
- Unchanged frames, random inputs, full-frame changes, and noisy moving shapes.
- Isolated impulses removed and a known hole filled.
- A translated rectangle yields two changed strips; unchanged overlap stays zero.
- Reusing the prediction tail with the host graph evaluator disabled.

The batch runtime tests cover invalid indices, duplicates, ordering, empty
inputs, wrong byte counts, decoder diagnostics, output mapping, and distinct
jobs separated by IDRs. Additional direct NVDEC checks rejected ten malformed
selection/count/dimension cases. NVDEC checks actual dimensions on every output;
software compatibility paths check byte counts and trust the caller's shape.

## Hardware and timing

The GStreamer helper uses `nvv4l2decoder`. It preserves exact full-range
surface formats while the VIC copies selected outputs into a CPU-readable
layout. No software decode fallback or host mask computation is used.

Each mode has one untimed warmup and five timed repetitions, rotating mode order.
All four modes process the same streams and check exact bytes on every run.
All-picture modes compare every raw byte; selective modes compare the final
I420 picture for every job. A separate full run supplies the capture planes.

| Mode | Sessions | Decoded / copied pictures | Median | Min–max |
| --- | ---: | ---: | ---: | ---: |
| Restart, all outputs | 24 | 960 / 960 | 11,805.73 ms | 11,632.67–11,929.42 ms |
| Restart, final masks | 24 | 960 / 24 | 11,701.41 ms | 11,622.82–11,836.23 ms |
| Batch, all outputs | 1 | 960 / 960 | 4,911.26 ms | 4,887.00–5,069.11 ms |
| Batch, final masks | 1 | 960 / 24 | 4,986.57 ms | 4,962.62–5,017.74 ms |

Timing includes concatenation and file I/O, process/pipeline startup, parsing,
all decoding, selected surface copies, CPU readback, diagnostic checks, and
byte comparison. SSH/network, compilation, PCM input packing, reference loading,
and visualization are excluded. Inputs are preconstructed; there is no live
camera capture or host feedback loop. The batch is finite, not a persistent
interactive submission service. Selective mode skips only external copies,
never the intermediate decoding required by subsequent arithmetic.

The batch/all median is 2.40× faster than restart/all. Batch/final is 2.37×
faster than restart/all and copies 40× fewer bytes. Selective batch readback
is slightly slower than full readback in this run; no latency improvement is
claimed for that change. At 40 decoded pictures per mask, these results do not
establish real-time video processing or a GPU-kernel speedup.

The saved local FFmpeg comparison has three repetitions and medians of 920.05 ms
for restart/all and 120.39 ms for batch/all. It ran on a different host, so it is
not a controlled hardware comparison. Software "selection" happens after full
decode/readback; its report correctly records all 960 copied pictures.

## Saved evidence

- [NVDEC results](assets/nvdec-results.json): raw/output hashes, backend
  provenance, copy counts, and every timing sample.
- [FFmpeg results](assets/ffmpeg-results.json): local runtime comparison.
- [Bundle manifest](assets/bundle.json): stream hashes, expected outputs,
  per-job packing observations, and local decoder validation.
- [Program manifest](assets/program.json): graph, ranges, reference dependencies,
  instructions, and physical output picture indices.
- [Source provenance](assets/source-provenance.json): historical source hashes from
  before redaction, associated with the recorded run. Historical host/platform metadata and identifying
  captions have been removed; measurements and decoded-pixel hashes are
  unchanged. Earlier commit IDs were not retained after the history rewrite.
  Future preparations record source hashes and dirty-tree status directly.
- [Executable stream](assets/motion.h264): all 24 jobs, with no expected output
  images supplied to the decoder.
- [Capture renderer](capture.py): verifies all 120 raw plane hashes before
  creating PNG/GIF/MP4. GIF/MP4 playback at 12 fps is presentation timing only.

Raw PGMs and full reference buffers are generated under ignored `out/` and can
be reproduced with the README commands. Report paths refer to those generated
directories rather than files included in the repository.
