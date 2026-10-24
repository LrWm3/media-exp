# Explicit H.264 levels

`Writer(..., level_idc=31)` remains the default. `level_idc=41` explicitly
enables High level 4.1. Other identifiers are rejected. For accepted
level-3.1 inputs, encoded bytes remain unchanged; selecting 4.1 changes the SPS level
byte, while prediction syntax remains the same.

```python
writer = Writer(256, 256, slice_mode="row", level_idc=41)
first = writer.input_luma(first_image, 0)
second = writer.input_luma(second_image, 1)
```

The writer uses 8-bit 4:2:0 High profile, progressive frames, sixteen declared
reference pictures, fixed integer frame rates from 1 through 60, and a
16-pixel zero guard on every side. Limits therefore apply to the complete
coded image, including those guards. Selecting 4.1 does not enable arbitrary
image sizes or automatically select a decoder.

## Checked limits

Values come from [ITU-T H.264 (08/2024), Annex A, Tables A-1, A-2 and A-4](https://www.itu.int/rec/T-REC-H.264-202408-S/en).

| Limit | High 3.1 | High 4.1 |
| --- | ---: | ---: |
| Maximum macroblocks per second (`MaxMBPS`) | 108,000 | 245,760 |
| Maximum macroblocks per picture (`MaxFS`) | 3,600 | 8,192 |
| Maximum decoded-picture storage in macroblocks (`MaxDpbMbs`) | 18,000 | 32,768 |
| Minimum compression ratio (`MinCR`) | 4 | 2 |
| Slice-rate divisor (`SliceRate`) | 60 | 24 |
| Maximum VCL bitrate | 17,500,000 bit/s | 62,500,000 bit/s |
| Maximum NAL bitrate | 21,000,000 bit/s | 75,000,000 bit/s |
| Maximum VCL coded-picture buffer | 17,500,000 bits | 78,125,000 bits |
| Maximum NAL coded-picture buffer | 21,000,000 bits | 93,750,000 bits |

Both coded macroblock dimensions must have their square no greater than
`8 * MaxFS`. Picture size, processing rate and the sixteen-picture DPB
declaration are checked independently. Prediction slice counts must satisfy
`slices * fps * SliceRate <= MaxMBPS`, including extra splits introduced by
per-block predictions. The initial PCM picture has one slice, which fits the
initial-picture slice limit for both supported levels.

Both levels limit horizontal luma motion to quarter-sample components
`[-8192, 8191]` and vertical motion to `[-2048, 2047]`. The separately declared
SPS motion bound can be tighter. Predictions use only complete 16×16 blocks
with one or two motion vectors, and no residuals; PCM macroblock syntax fits
the common `128 + RawMbBits` bound.

For coded picture size `M` macroblocks, the first access unit (AU) is limited
to `floor(384 * max(172*M, MaxMBPS) / (172*MinCR))` NAL-unit bytes.
Each later AU is limited to `floor(384 * MaxMBPS / (fps*MinCR))` bytes.
These implement A.3.3(i,j) using `fR = 1/172`, fixed frame intervals and no
delayed initial removal. Counts include escaped NAL headers and payloads;
four-byte Annex B start codes are excluded from these two limits.

The writer also checks each complete AU against the inferred VCL and NAL CPB
capacities. VCL counts include only slice NAL units. NAL/byte-stream checks
include parameter sets and conservatively count Annex B start codes too.

Explicit level 4.1 additionally requires each AU to fit one frame interval
at both inferred maximum bitrates: `VCL_bytes * 8 * fps <= 62,500,000` and
`byte_stream_bytes * 8 * fps <= 75,000,000`. This sufficient stateless bound
accepts a narrower subset than full level-4.1 HRD conformance: valid bursts
that use buffer capacity across several frame intervals can be rejected.
This is not a general HRD simulator. Level 3.1 retains its historical burst
behavior and does not gain this additional per-AU rate cap; complete
level-3.1 sequence HRD validation remains outside this writer.

## Why 256×256 needs explicit 4.1

A 256×256 active image becomes 288×288 coded pixels, or 324 macroblocks.
At 30 fps this uses 9,720 macroblocks/s and 5,184 macroblocks of declared
DPB storage. Row predictions use 50 slices; macroblock predictions use 324.
Both fit level 4.1 at 30 fps, including its 10,240-slices/s bound.

The level-3.1 initial AU limit is 60,279 bytes. The all-zero 256×256 fixture
requires 166,571 NAL-unit bytes and is rejected. Lowering fps does not change
that initial limit. Level 4.1 raises it to 274,336 bytes. At 30 fps its later
AU limit is 1,572,864 bytes, and its stricter per-AU VCL rate budget is
260,416 bytes. Even a generic 3/2 emulation-prevention expansion bound for
this PCM geometry stays below 188,000 bytes. Thus arbitrary byte-valued
256×256 inputs fit these budgets at 30 fps; higher frame rates still undergo
the same checks and can reject an input.

`python3 -m unittest decoder_core.test_levels` checks exact budget boundaries,
level rejection, historical default bytes and full I420 decoding. Decoder
fixtures cover zero, 255, alternating and full-byte values; two input AUs;
average and motion-dependent difference predictions; and both slice modes.
Each available local FFmpeg, OpenH264 and VA-API backend must match every
picture byte, including guards and neutral chroma. These are correctness
checks, not throughput measurements.
