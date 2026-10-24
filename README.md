# Media experiments

- [01 — Conway's decoder of Life](01-conways-decoder-of-life/README.md): a generated
  H.264 bitstream makes software and hardware video decoders compute Conway's
  Game of Life using motion compensation, weighted prediction, and clipping.
- [02 — Maze flood-fill decoder solver](02-maze-flood-fill-decoder-solver/README.md):
  the decoder expands reachability through a maze; decoded arrival times provide
  shortest distances and paths while walls block propagation.
- [03 — Denoising decoder](03-denoising-decoder/README.md): a PyTorch-trained binary
  CNN compiles into H.264 prediction operations, including signed 3×3 filters,
  binary feature maps, and a streaming 1×1 output layer.
- [04 — Binary MNIST classification decoder](04-binary-mnist-classification-decoder/README.md):
  a trained binary network classifies handwritten 3s versus 8s using decoder
  pooling, signed arithmetic, and thresholds; 94.96% accuracy on the official
  two-class test subset.
- [05 — Python H.264 compiler](05-h264-compiler/README.md): a shared symbolic image
  language with exact arithmetic, shift fusion, common-expression elimination,
  reference scheduling, and inspectable instruction manifests. Its
  [native-operation demos](05-h264-compiler/native_ops/README.md) add multiple
  input images, saturating arithmetic, block gathers, and quarter-pixel interpolation,
  with measured input-loading and decoder costs.
- [06 — Motion-mask decoder](06-motion-mask-decoder/README.md): frame differences,
  thresholding, and binary mask cleanup; persistent NVDEC batches and selective
  output readback, with exact captures and measured costs.
- [07 — Packed denoising decoder](07-packed-denoising-decoder/README.md): independent
  neural feature calculations share picture tiles, reducing 52 pictures to 17;
  unchanged weights, exact outputs, and paired NVDEC measurements.
- [08 — Decoder computation limits](08-decoder-compute-limits/README.md): proved
  minima for all 16 two-input Boolean functions, synthesis/scheduling tradeoffs,
  measured decoder costs, and exact denoising alongside CUDA workloads.

Compiler variants: [Life](01-conways-decoder-of-life/compiled/README.md) ·
[Maze](02-maze-flood-fill-decoder-solver/compiled/README.md) ·
[Denoising](03-denoising-decoder/compiled/README.md) ·
[MNIST](04-binary-mnist-classification-decoder/compiled/README.md).

The experiments use the shared [H.264 writer and decoder tools](decoder_core).

[Multi-codec compute IR](codec_compute/README.md): an isolated extension preserves
the H.264 compiler, adds resident HEVC weighted-prediction programs and an AV1
copy/translation/average target, and separates exact image semantics from
codec-specific instruction selection and reference scheduling. Includes real
decoder validation, explicit unsupported operations, and reusable computation tails.

[Automatic computation synthesis](codec_compute/synthesis/README.md): generic
bounded search discovers min/max and Boolean AND/OR/XOR implementations from
exact codec primitive descriptions, retains exhaustive scalar verification
records, and emits real H.264/HEVC/AV1 programs for successful searches.

[NVDEC validation](05-h264-compiler/native_ops/nvdec/README.md):
34 cases and 6,116 decoded pictures matched exactly on NVDEC, including all
four compiled experiments. Captures, timings, and reproduction commands included.

[Compiler optimization results](05-h264-compiler/optimizations/README.md):
Boolean and threshold lowering, grouped prediction slices, retained reference
slots, and exact comparisons across all six experiments.

[Exact instruction search](05-h264-compiler/optimizations/exact-search/README.md):
proved small-domain replacements reduce Life from 682 to 402 pictures and
denoising from 58 to 52, with decoder captures and paired NVDEC measurements.

[Maze kernel and readback improvements](02-maze-flood-fill-decoder-solver/compiled/optimizations/README.md):
265 → 214 decoded pictures, 265 → 54 NVDEC copies, and a measured 1.12× speedup
while preserving every flood state and shortest distance.
