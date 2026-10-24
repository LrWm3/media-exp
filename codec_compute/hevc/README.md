# HEVC weighted-prediction backend

This backend emits real HEVC Annex B streams. FFmpeg's HEVC decoder computes
weighted predictions. No host evaluator supplies arithmetic results to the
encoder. Compilation creates an all-zero template using libx265, retains its
CABAC skip payload, and changes the slice prediction weights. Consequently
`Kernel.tail` depends on shape and operation only, and is reused byte for byte
for every input upload.

```python
from codec_compute.hevc import compile_binary

kernel = compile_binary(32, 32, left_weight=2, right_weight=-2)
stream = kernel.assemble(bytes([200]) * 1024, bytes([70]) * 1024)
result = kernel.decode(bytes([200]) * 1024, bytes([70]) * 1024)
assert result == bytes([130]) * 1024
```

The input loader uses libx265 lossless encoding. Unary kernels upload one IDR
picture and append a P-picture computation. Binary kernels feed `A,A,B` into a
fixed one-B-frame encoder schedule, retain uploaded `A` (IDR, POC 0) and `B`
(P, POC 2), discard the duplicate A's B-picture, and append the compiled B-picture
(POC 1). This duplicate is input-upload work; no computed result is uploaded.
Displayed frames are `[A, result]` or `[A, result, B]`, respectively.

Exact luma semantics, with arithmetic right shifts:

- Unary: `clip_u8(((w*x + round) >> d) + offset)`, where `round=0` for `d=0`
  and `round=2**(d-1)` otherwise.
- Binary: `clip_u8(((wa*a + wb*b + 2**d) >> (d+1)) + offset)`.

`(wa,wb,d)=(2,2,0)` implements saturating add; `(2,-2,0)` saturating subtract;
`(1,1,0)` rounded average. Chroma is neutral 128 and passes through unchanged.

Constraints: 8-bit luma, width and height multiples of 8 in `[16,512]`, one
slice, zero motion, disabled temporal MVP, SAO and deblocking. Weight delta
`w-2**d` must fit `[-128,127]`; offset must fit `[-128,127]`; `d` is `[0,7]`.
This is a constrained software-decoder implementation, not a claim about
hardware acceleration or general HEVC programmability. Requires `ffmpeg` with
`libx265`, `trace_headers`, and the HEVC decoder. Parameter-set equality checks
reject incompatible encoder configurations.

Validation: `python3 -m unittest codec_compute.tests.test_hevc -v`. Tests decode
actual streams, cover constants, ramps, checkerboards and seeded random inputs,
check display-order uploads as well as computed outputs, and verify reusable
tails for multiple shapes and multi-CTU pictures.

Implementation references:

- [x265 lossless coding](https://x265.readthedocs.io/en/stable/lossless.html).
- [x265 encoder options](https://x265.readthedocs.io/en/stable/cli.html).
- [FFmpeg HEVC slice-header parsing](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/hevc/hevcdec.c).
