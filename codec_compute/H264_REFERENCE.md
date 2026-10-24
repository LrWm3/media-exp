# H.264 reference implementation

The existing compiler in `05-h264-compiler/h264` remains unchanged. This experiment
adds target adapters beside it. Its original DSL and examples remain available.

## Computation

The immutable `Image` graph carries unsigned eight-bit values and inclusive range
proofs. Strict affine operations reject possible overflow. Saturating operations
clip after the complete sum. Python loops build finite graphs; input-dependent
Python control flow is rejected.

For integer samples, weighted P prediction computes
`clip8(round_half_up(w*x / 2**d) + offset)`. Weighted B prediction computes
`clip8(round_half_up((w0*x+w1*y) / 2**(d+1)) + round_half_up((o0+o1)/2))`.
Rounding uses integer arithmetic, including negative values. Equal source offsets
and doubled B weights implement exact binary affine operations. Boolean OR uses
rounded averaging of 0/1 samples; AND uses `clip8(x+y-1)`.

Fractional sampling means H.264's specified six-tap interpolation, with its exact
rounding and clipping. It cannot silently become another codec's interpolation.
Integer shifts use zero borders. Intermediate crop and saturation boundaries are
observable and must survive transformations.

## Realization and limits

- Input upload uses lossless I_PCM pictures. Named inputs contain literal bytes;
  the historical unnamed binary-input API instead uploads visible 0/255 and
  normalizes inside the decoder.
- Prediction pictures have no residual. A reusable prediction tail depends on
  program parameters, not actual input pixels.
- The implementation uses a 16-pixel zero guard, neutral chroma, and disabled
  deblocking. Shapes are multiples of 16 and obey the writer's level limits.
- Retained long-term reference slots remove picture-age restrictions. The
  scheduler currently permits at most 15 live values before an instruction;
  it has no general spilling or recomputation.
- Weights, offsets, quarter-pixel motion limits, reference management, macroblock
  layout, and entropy syntax are target details. They do not define portable IR
  legality.
- `backend="ffmpeg"` selects a decoder, not a codec target.

## Baseline on this branch

Before integration, 33 tests passed in 5.877 seconds:

```sh
PYTHONPATH="$PWD:$PWD/05-h264-compiler" python -m unittest discover \
  -s 05-h264-compiler -p test_native_ops.py
# The complete recorded baseline also ran test_compiler, test_interpolation,
# and test_schedule using unittest's module selection.
```

Available decoder paths exercised by those tests: FFmpeg, OpenH264, and VA-API.
Coverage includes every pair of u8 saturating-add operands, threshold cutoffs
outside and throughout the byte range, all fractional phases, retained
references, and a 1,500-step graph. NVDEC is unavailable on this host. Earlier
repository NVDEC results are historical evidence, not validation of new targets.

See `05-h264-compiler/DESIGN.md`, `h264/{graph,lower,schedule,verify}.py`,
and `decoder_core/bitstream.py` for the executable reference.
