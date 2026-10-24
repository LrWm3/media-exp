# Resident IDR fence

`neutral_idr(writer)` emits one discarded picture with every luma and chroma
sample equal to 128. Its shape, SPS and PPS match the computation writer.
The ordinary program's zero guard does not apply to this runtime picture.

The construction follows [H.264 (08/2024)](https://www.itu.int/rec/T-REC-H.264-202408-S/en):

- Table 7-11: macroblock type 3 selects Intra16 DC with zero luma AC and chroma coded patterns.
- Clauses 7.3.5 and 7.3.5.3.1: chroma DC mode, a zero QP delta, and the luma DC residual block remain present.
- Clause 9.2.1 and Table 9-5: all neighboring AC counts are zero, so `nC=0`; the single bit `1` encodes zero coefficients.
- Clause 8.3.3.3 and chroma DC prediction: unavailable edges start at 128; constant neighboring samples preserve 128 throughout the picture.
- Clause 7.4.3: consecutive IDR pictures require different `idr_pic_id` values. Ordinary inputs use 0 and fences default to 1. Tests with two consecutive fences use 1 and 2.

The fence sets `no_output_of_prior_pics_flag=0`. With fixed geometry and DPB
parameters, pending pictures remain eligible for output, while the IDR clears
the previous reference state. A following job starts with its own input IDR.
The fence is additional decoded work, not a graph output or a reusable input.

Local tests compare every decoded I420 byte using FFmpeg, VAAPI and OpenH264,
when available, at active sizes 128x128 (High 3.1) and 256x256 (High 4.1).
They cover alternating fence IDs and changing PCM/P/B jobs separated by fences.
They establish picture correctness, not persistent-session timing or NVDEC liveness.

GStreamer OpenH264 1.24.2 with libopenh264 2.4.1 locally failed to drain a
single-picture stream before producing its first output. This also happened
with an ordinary PCM IDR. Adding a second valid picture released both pictures;
the first tiny IDR decoded exactly. A separate resident NVDEC investigation
reported a decoder failure for an earlier P-picture delimiter; that hardware
result is not reproduced by this local test suite.
