# Decoder checks and measured execution costs

The [frozen bundle index](assets/bundle.json) records 44 cases: all 16 functions
in scalar and packed modes, plus 12 dependent-update microbenchmarks. The
[NVDEC report](assets/nvdec-results.json) contains individual timings, counters,
output hashes, and the hash of that exact index.

## Correctness

FFmpeg, OpenH264, VA-API, and NVDEC agree on **2,618 complete decoded pictures**,
**25,045,248 I420 bytes per backend**. Comparisons cover input pictures,
intermediates, outer guards, unused tiles, chroma, and outputs. Every final
output also matches an independent Boolean truth-table oracle. Native outputs
are 0/1, with no display normalization inside the measured stream.

The certificate covers all 64 input/function truth-table combinations. Program
tests additionally cover changing input pixels with an identical cached
prediction tail, reference lifetimes across long recurrences, aliases,
constants, and rejection of corrupted output. Runner tests reject changed
artifacts and input seeds even when an altered file's checksum is updated.
The [proofs](PROOFS.md) state the exact scope of the minimality claims.

The complete experiment test suite passes **31 tests**, including the joint
schedule comparison and concurrency-runner checks. Shared compiler/core sources
remain unchanged in this experiment.

The bundle was generated from commit
`d683dc4868fc63ddc4773fe93aec2f8d4cf7b357` with a clean working tree. Its index
records hashes of all relevant sources. Later additions provide the independent
schedule comparison and documentation; they do not change the measured streams.

## Timing boundary

NVDEC runs each stream in a new helper process and decoder session. Each
case/readback mode receives one warmup and five timed runs; their order rotates.
Timing includes stream assembly and file I/O, process/decoder startup, parsing,
decoding, surface copies, diagnostics, byte-count checks, and frame slicing.
It excludes compilation, PCM input packing, loading expected references,
pixel equality checks, network transfer, and capture rendering.

`outputs` copies only the final requested picture. `all` copies every picture.
Both modes decode the same complete stream. No clock or power settings are
changed. Samples have visible variation; some full-readback medians are lower
than selected-readback medians. That is not evidence that copying more data is
intrinsically faster.

## 128-round execution results

One round applies `A_next = f(A, original_B)` at every logical cell. Total
pictures include two input pictures. Update rates use
`rounds × logical_width × logical_height / elapsed_seconds`.

| Kernel | Board | Coded size | Pictures | Final readback | All readback | Million updates/s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| NAND | 16×16 | 48×48 | 130 | 466.24 ms | 473.69 ms | 0.070 |
| NAND | 32×32 | 64×64 | 130 | 479.69 ms | 476.51 ms | 0.273 |
| NAND | 64×64 | 96×96 | 130 | 484.21 ms | 493.00 ms | 1.083 |
| Scalar XOR | 16×16 | 48×48 | 386 | 889.52 ms | 889.06 ms | 0.037 |
| Scalar XOR | 32×32 | 64×64 | 386 | 854.88 ms | 891.01 ms | 0.153 |
| Scalar XOR | 64×64 | 96×96 | 386 | 954.57 ms | 943.68 ms | 0.549 |
| Packed XOR | 16×16 | 80×48 | 258 | 638.16 ms | 642.69 ms | 0.051 |
| Packed XOR | 32×32 | 112×64 | 258 | 643.78 ms | 644.72 ms | 0.204 |
| Packed XOR | 64×64 | 176×96 | 258 | 837.72 ms | 756.07 ms | 0.626 |

These recurrences deliberately retain dependent instructions to measure
execution. Their mathematical result may need much less work: for example,
even XOR iteration counts return the original A. They cannot establish a
useful application speedup or be converted into equivalent GPU operations.

At 32×32, NAND spends **4 decoded computation luma samples per logical
update**, scalar XOR spends **12**, and packed XOR spends **14**. These include
guards and empty tiles but exclude the two input pictures. Packed XOR therefore
improves time despite increasing decoded computation samples by 16.7%.
Across the 64×64 cases, observed total luma rates are 2.47 million/s for NAND,
3.73 million/s for scalar XOR, and 5.20 million/s for packed XOR. They measure
this small-stream runtime, including startup; they are not hardware limits.

## Approximate incremental cost

At fixed 32×32 geometry, subtracting the 32-round median from the 128-round
median approximately cancels startup. With final-picture readback:

| Kernel | 32 rounds | 128 rounds | Difference | Incremental million updates/s |
| --- | ---: | ---: | ---: | ---: |
| NAND | 335.39 ms | 479.69 ms | 144.31 ms | 0.681 |
| Scalar XOR | 438.90 ms | 854.88 ms | 415.99 ms | 0.236 |
| Packed XOR | 393.75 ms | 643.78 ms | 250.03 ms | 0.393 |

This is a difference of pipeline medians, not isolated GPU execution time or
a fitted steady-state limit. Input packing and first tail generation have
separate host measurements in the bundle. Slice counts, reference slots,
bitstream bytes, and copied/returned bytes remain available in the raw records.
The useful-work comparison is the [denoiser concurrency test](CONCURRENCY.md).
