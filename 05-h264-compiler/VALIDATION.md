# Validation

The [four-picture maze and readback report](../02-maze-flood-fill-decoder-solver/compiled/optimizations/README.md)
extends the local regression run to 184 passing tests. The shared compiler has
78 tests, including selected-picture ordering, output aliases, missing-picture
rejection and guard/chroma/range validation. Ten paired maze fixtures passed
full and selected NVDEC byte comparisons; the demo uses 214 decoded pictures
and 54 copied pictures in the new mode.

The latest [exact-search report](optimizations/exact-search/README.md) records
148 passing tests and 42 saved cases verified on FFmpeg, OpenH264, VA-API and
NVDEC: 4,855 pictures and all 323 named output planes. Paired measurements
compare the same compiler with only exact instruction search enabled/disabled.

The [compiler optimization report](optimizations/README.md) records a 124-test
local regression run, exact comparisons of 42 saved demo executions, and fresh
decoder measurements.
The sections below preserve the earlier compiler and native-operation baselines.

Recorded on 2026-10-23 with Python 3.12.13, FFmpeg 6.1.1, GStreamer 1.24.2,
OpenH264, and Intel VA-API through `vah264dec`. Neural checkpoint comparisons
used PyTorch 2.5.1+cpu. Original model weights were not retrained or modified.

## NVDEC

The subsequent [NVDEC run](native_ops/nvdec/README.md) passed
**34 cases / 6,116 pictures /
101,193,216 raw I420 bytes**. Every byte matched FFmpeg; all 300 named outputs
also matched independent integer graphs. Coverage includes the six native
demos, all 65,536 byte-addition pairs, fractional phases and block gathers,
signed biprediction, long-lived references, and all four committed compiled
experiment streams (eight denoiser images and 16 MNIST digits).

The new readback helper preserves full-range surface formats during VIC layout
copy and copies actual NV12 planes to I420. This avoids the observed 0–255 to
16–235 conversion in the initial `nvvidconv` readback attempt. Streams and the
NVIDIA decoder remain unchanged. Unknown diagnostics, decoder errors, and
concealed blocks fail the run.

After adding the backend, **126 local regression tests passed with no skips**:
the 116 below plus ten NVDEC diagnostic/backend tests. The same 116 suites
were rerun, including all original and compiled experiments. The full neural
accuracy datasets were not rerun. [Results and timing samples](native_ops/nvdec/results.json)
and [actual NVDEC captures](native_ops/nvdec/README.md#what-passed) are saved.
Desktop CUVID and SDK 13.1 direct-array output remain untested.

```sh
python3 -m unittest decoder_core.test_decode -v
```

## Native operations, v0.2

**116 tests passed, with no skips**, after adding native image operations and
replacing bit-at-a-time PCM packing with byte packing:

| Suite | Tests |
| --- | ---: |
| Compiler: original core + native arithmetic + interpolation | 30 |
| Shared writer: syntax, input retention, motion, byte equivalence | 10 |
| Four compiled experiment variants | 34 |
| Four original experiments | 42 |

The new arithmetic suite exhausts **all 65,536 unsigned-byte pairs** for
saturating addition on FFmpeg, OpenH264, and Intel VA-API. It also checks signed
weights and offsets, two-input lifetime refreshes, 15 input pictures, seed aliases,
prediction reuse without host inference, and explicit rejection of unsupported
forms. The addition needs two input pictures and exactly one weighted B-picture.

Interpolation tests cover all 16 quarter-pixel phases, negative coordinates,
edges, diagonal intermediate precision, per-block fractional gathers, and nested
filtering with optimization enabled and disabled. Every decoder agrees with an
independent integer oracle. A separate review exercised signed fractional
biprediction, three named inputs, and gathers separated by long reference chains.

Byte packing preserves legacy seed hashes and matches the former encoder
byte-for-byte for four shapes, four input patterns, IDR/non-IDR pictures, and
frame-number wrap boundaries. Separate encoding-only measurements for two 64×64
random inputs fell from 26.56 ms to 1.07 ms (five repetitions of 20 calls per
implementation). This improves host serialization, not decoder arithmetic.

Six [native demos](native_ops/README.md) verify **172,032 named-output pixels**
across all three decoders. Their committed manifests, raw-plane PNGs, annotated
hardware captures, and [results](native_ops/results.json) identify actual output
frames and hashes. The [benchmark](native_ops/BENCHMARK.md) includes both
separate-process and one-session fresh-input baselines. For 32 additions at
64×64, Intel VA-API took 64.46 ms with fresh pairs in one session and 63.60 ms
with resident references. These medians include startup, all-picture readback,
and validation; the small difference does not establish a hardware speedup.
Input packing for the complete workloads was 44.71 ms versus 1.39 ms.

NVDEC validation is recorded above. Direct-array output remains untested.
The full 1,984-image MNIST accuracy and 256-image denoiser reports below are the
previous v0.1 runs; this update reran their regression suites rather than those
full datasets.

Run from the repository root:

```sh
python3 -m unittest discover -s 05-h264-compiler -v
python3 -m unittest decoder_core.test_native_bitstream -v
```

Run each original/compiled experiment's suite from its own directory using
`03-denoising-decoder/.venv/bin/python -m unittest -v` (absolute interpreter path).

## Initial v0.1 tests

**48 new tests passed, with no skips** in the recorded environment:

| Suite | Tests | Checks |
| --- | ---: | --- |
| Compiler core | 14 | Exact arithmetic, thresholds, boundaries, liveness, manifests, reuse, safe failure |
| Compiled Life | 9 | All 512 local configurations, full generations, original-stream parity |
| Compiled maze | 14 | All 243 local configurations, every state versus BFS, distances, original parity |
| Compiled denoiser | 5 | All 512 binary patches, every feature/output, original checkpoint |
| Compiled MNIST | 6 | Pooling/hidden/class bits, tile isolation, original checkpoint |

Run the core suite here with `python3 -m unittest -v`. Variant READMEs give
their integration commands. Use `03-denoising-decoder/.venv/bin/python`, or an
equivalent environment, for checkpoint tests. Missing optional dependencies are
reported as skips rather than successes.

The core exhausts all 256 byte values at all 259 integer cutoffs from −1 through
257 on FFmpeg, OpenH264, and VA-API. Boundary tests preserve cropping between
opposing shifts and distinguish shifting before versus after adding bias.
A 1,500-step graph checks iterative traversal; a separate long-lived input
requires reference refreshes. An excessive live set fails compilation.

An independent review additionally decoded 80 signed-affine cases with large
shifts and biases through all three backends. It found and fixed a save-path
collision: a `.json` stream path now fails before overwriting any file.
No computational mismatch was found.

## Recompiled workloads

The Life capture contains 40 generations of a 32×32 r-pentomino; all 41 states
match its oracle. The maze capture contains 52 steps, reaches 359 cells, leaves
9 sealed cells unreachable, reaches its goal in 50 hops, and converges at step 51.
All three decoders produce identical raw bytes for these committed streams.
See [Life validation](../01-conways-decoder-of-life/compiled/VALIDATION.md) and
[maze validation](../02-maze-flood-fill-decoder-solver/compiled/VALIDATION.md).

The denoiser processes all 256 original held-out images. Every decoded feature
and output bit matches both the original integer model and PyTorch checkpoint.
Pixel error remains 3.293228%, from 15.071487% noisy input. The compiler emits
74 pictures/image instead of 98 by sharing repeated convolutions. The
[held-out report](../03-denoising-decoder/compiled/assets/held-out.json) records
hashes and metrics; the hardware capture uses eight images.

MNIST processes all 1,984 official test 3s and 8s in original order. Every pooled
cell, hidden bit, and prediction matches the integer model and PyTorch checkpoint.
Every prediction/hidden record also matches the original CSV. Accuracy remains
**1,884 / 1,984 = 94.9597%**:

| True class | Predicted 3 | Predicted 8 |
| --- | ---: | ---: |
| 3 | 952 | 58 |
| 8 | 42 | 932 |

The compiler emits 1,826 pictures per eight-digit batch versus 1,802 originally.
Its full test stream is 522,182,284 bytes, retained under ignored `compiled/out/`.
See the [full report](../04-binary-mnist-classification-decoder/compiled/assets/test-set.json)
and [original-result comparison](../04-binary-mnist-classification-decoder/compiled/assets/original-parity.json).
The hardware capture uses the 16 checked-in digits; the full accuracy run uses FFmpeg.

## Scope

The results establish exact execution for the supported language and tested
programs. They do not certify every possible graph or decoder implementation,
or make recognition perfect. Unsupported arithmetic and excess reference
pressure fail compilation.

The generic compiler increases picture count for Life, maze, and MNIST; repeated
filters make the denoiser smaller. Timings are local observations, not controlled
comparisons. Writer serialization is lazy and may be charged to first encoding
rather than graph compilation. Captures enlarge and annotate actual decoded
values; executable streams contain the underlying computation pictures.
