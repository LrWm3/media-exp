# Explicit H.264 levels

`Writer(..., level_idc=31)` remains the default. `level_idc=41` and `51`
explicitly enable High levels 4.1 and 5.1. Other identifiers are rejected. For accepted
level-3.1 inputs, encoded bytes remain unchanged; selecting a higher level changes the SPS level
byte, while prediction syntax remains the same.

```python
writer = Writer(256, 256, slice_mode="row", level_idc=41)
first = writer.input_luma(first_image, 0)
second = writer.input_luma(second_image, 1)
```

The writer uses 8-bit 4:2:0 High profile, progressive frames, sixteen declared
reference pictures, fixed integer frame rates from 1 through 60, and a
16-pixel zero guard on every side. Limits therefore apply to the complete
coded image, including those guards. Selecting a higher level does not enable arbitrary
image sizes or automatically select a decoder.

## Checked limits

Values come from [ITU-T H.264 (08/2024), Annex A, Tables A-1, A-2 and A-4](https://www.itu.int/rec/T-REC-H.264-202408-S/en).

| Limit | High 3.1 | High 4.1 | High 5.1 |
| --- | ---: | ---: | ---: |
| Maximum macroblocks per second (`MaxMBPS`) | 108,000 | 245,760 | 983,040 |
| Maximum macroblocks per picture (`MaxFS`) | 3,600 | 8,192 | 36,864 |
| Maximum decoded-picture storage in macroblocks (`MaxDpbMbs`) | 18,000 | 32,768 | 184,320 |
| Minimum compression ratio (`MinCR`) | 4 | 2 | 2 |
| Slice-rate divisor (`SliceRate`) | 60 | 24 | 24 |
| Maximum VCL bitrate | 17,500,000 bit/s | 62,500,000 bit/s | 300,000,000 bit/s |
| Maximum NAL bitrate | 21,000,000 bit/s | 75,000,000 bit/s | 360,000,000 bit/s |
| Maximum VCL coded-picture buffer | 17,500,000 bits | 78,125,000 bits | 300,000,000 bits |
| Maximum NAL coded-picture buffer | 21,000,000 bits | 93,750,000 bits | 360,000,000 bits |

These values were checked against the official V15 PDF dated 08/2024,
Tables A-1, A-2 and A-4 on printed pages 298, 302 and 303. The PDF has
15,371,015 bytes and SHA-256
`647d6bb2f7a62d221cd5e31da9a948e2dde32f9ac008a548f98348c838bd389d`.
Table A-1 gives level-5.1 `MaxBR=240000` and `MaxCPB=240000`; Table A-2
applies High-profile factors 1250 for VCL and 1500 for NAL.

Both coded macroblock dimensions must have their square no greater than
`8 * MaxFS`. Picture size, processing rate and the sixteen-picture DPB
declaration are checked independently. Prediction slice counts must satisfy
`slices * fps * SliceRate <= MaxMBPS`, including extra splits introduced by
per-block predictions. The initial PCM picture has one slice, which fits the
initial-picture slice limit for all supported levels.

All supported levels limit horizontal luma motion to quarter-sample components
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

Explicit levels 4.1 and 5.1 additionally require each AU to fit one frame interval
at both inferred maximum bitrates: `VCL_bytes * 8 * fps <= 1250 * MaxBR` and
`byte_stream_bytes * 8 * fps <= 1500 * MaxBR`. This sufficient stateless bound
accepts a narrower subset than full HRD conformance: valid bursts
that use buffer capacity across several frame intervals can be rejected.
This is not a general HRD simulator. Level 3.1 retains its historical burst
behavior and does not gain this additional per-AU rate cap; complete
level-3.1 sequence HRD validation remains outside this writer.

## Why 256×256 needs explicit 4.1

A 256×256 active image becomes 288×288 coded pixels, or 324 macroblocks.
At 30 fps this uses 9,720 macroblocks/s and 5,184 macroblocks of declared
DPB storage. The default Writer row policy uses 50 slices; eligible zero-guard
merging uses 18. Macroblock predictions use 324 slices. These fit level 4.1 at
30 fps, including its 10,240-slices/s bound.

The level-3.1 initial AU limit is 60,279 bytes. The all-zero 256×256 fixture
requires 166,571 NAL-unit bytes and is rejected. Lowering fps does not change
that initial limit. Level 4.1 raises it to 274,336 bytes. At 30 fps its later
AU limit is 1,572,864 bytes, and its stricter per-AU VCL rate budget is
260,416 bytes. Even a generic 3/2 emulation-prevention expansion bound for
this PCM geometry stays below 188,000 bytes. Thus arbitrary byte-valued
256×256 inputs fit these budgets at 30 fps; higher frame rates still undergo
the same checks and can reject an input.

## Arbitrary 512×512 inputs with explicit 5.1

A 512×512 active image becomes 544×544 coded pixels, or 1,156 macroblocks.
At 30 fps it uses 34,680 macroblocks/s and 18,496 macroblocks of declared
DPB storage. The default Writer row policy uses 98 slices before field-induced
splits; eligible [zero-guard merging](GUARDS.md) reduces this to 34. Even 1,156
individual macroblock slices fit the 1,365-slice per-picture budget. Table A-4
requires progressive frames and `direct_8x8_inference_flag=1`
at level 5.1; the writer already sets both flags and uses 16×16 predictions,
which satisfy the 8×8 minimum bidirectional prediction block size.

Annex A.3 retains `fR=1/172` at level 5.1. The initial NAL-unit byte limit
is 1,097,346; the later limit at 30 fps is 6,291,456. The stricter per-AU
rate budgets are 1,250,000 VCL bytes and 1,500,000 byte-stream bytes.
PCM samples and macroblock syntax require about 446,225 unescaped bytes.
Allowing a full 3/2 emulation-prevention expansion and generous header and
parameter-set overhead still stays below 672,000 bytes for every possible
input sample pattern. Thus arbitrary byte-valued 512×512 inputs fit at 30 fps.
All actual emitted bytes are still checked; this does not promise larger
dimensions or arbitrary frame rates. Level 4.1's 274,336-byte initial limit
cannot hold this geometry's PCM input even before worst-case escaping.

Use `Writer(512, 512, slice_mode="row", level_idc=51)` directly or
`codec_compute.compile(kernel, width=512, height=512, h264_level=51)`.
Hardware decoder support must be checked separately from bitstream legality.

Opt-in [prediction slice bands](GUARDS.md#opt-in-prediction-slice-bands) can
reduce eligible prediction slice counts further: a 512×512 picture uses 34
slices with the default `prediction_slice_rows=1`, 9 with `4`, or 1 with `0`.
This groups coded macroblock rows only when zero-guard merging is enabled and
headers are uniform with zero offsets. Other predictions retain their existing
layout. Banding does not reduce coded dimensions, DPB use or input-upload
budgets; all actual emitted-byte limits and conservative constructor checks
still apply.

`python3 -m unittest decoder_core.test_levels codec_compute.tests.test_runtime`
checks exact budget boundaries,
level rejection, historical default bytes and full I420 decoding. Decoder
fixtures cover zero, 255, alternating and full-byte values at 256 and 512; two input AUs;
average and motion-dependent difference predictions; and both slice modes.
Each available local FFmpeg, OpenH264 and VA-API backend must match every
picture byte, including guards and neutral chroma. Public-IR tests also check
512×512 fractional warp/blend and every stage of a sixteen-round warp/blend
chain. These are correctness
checks, not throughput measurements.
