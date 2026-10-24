# Validation

Recorded on 2026-10-23 with Python 3.12.13, FFmpeg 6.1.1, GStreamer 1.24.2,
OpenH264, and Intel VA-API through `vah264dec`. Neural checkpoint comparisons
used PyTorch 2.5.1+cpu. Original model weights were not retrained or modified.

## Tests

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
