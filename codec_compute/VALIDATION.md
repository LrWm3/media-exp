# Validation record — 2026-10-24

Environment: Python 3.12.13; FFmpeg 6.1.1-3ubuntu5; pinned libaom serializer
v3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`). The AV1 serializer
was built using the documented bootstrap and its second invocation reused the
cached executable. Video decoders are unmodified.

## Automatic synthesis extension

The complete suite now passes **93 tests in 21.312 s, with no skips**. This adds
43 synthesis tests: 13 search, 17 independent verification, and 13 compiler/
software-decoder integration tests. This run used software decoders only;
hardware validation remains deferred for synthesis.

```sh
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.synthesis.demo
```

The generic engine discovers full-byte min/max and Boolean AND/OR/XOR for H.264
and HEVC, and Boolean OR for AV1. Every successful plan has an exhaustive scalar
record: 65,536 states for each byte extremum and four states for each Boolean
gate. The remaining four AV1 goals report `instruction_bound` for the initial
average-only library at three tree applications; this is not an impossibility
claim about AV1.

The integration suite decodes **all 65,536 byte pairs** for both synthesized
extrema on H.264 and HEVC, and all four Boolean pairs for the gates. AV1 OR
passes both libdav1d and libaom-av1. Other checks cover opaque callable goals
(including full-byte absolute difference), counterexample refinement, budget
exhaustion, descriptor-based arithmetic independent of evaluator overrides,
signed rounding, tampered-record rejection, composed Boolean ranges, existing
strict affine lowering, backend resource rejection after scalar success, and
changed inputs with no runtime synthesis or host computation.

The synthesis demo writes eleven streams and their manifests, plus a report
covering fifteen target/goal combinations, under
`codec_compute/artifacts/synthesis/`. Its per-stream software check uses one
16x16 pair; exhaustive decoded byte coverage belongs to the test suite.

| Discovered program | Target | Compute pictures | Prediction bytes |
| --- | --- | ---: | ---: |
| Min, max, or Boolean AND (each) | H.264 | 2 | 212 |
| Boolean OR | H.264 | 1 | 105 |
| Boolean XOR | H.264 | 3 | 318 |
| Min, max, or Boolean AND (each) | HEVC | 2 | 32 |
| Boolean OR | HEVC | 1 | 15 |
| Boolean XOR | HEVC | 3 | 47 |
| Boolean OR | AV1 | 1 | 27 |

Each program also loads two input pictures. Counts reflect this small grammar,
not optimal target costs. Scalar evidence and software-decoder evidence remain
separate in the report. See [synthesis details](synthesis/README.md).

## Original multi-codec milestone results

| Check | Result |
| --- | --- |
| Existing H.264 compiler/native/interpolation/scheduler baseline | 33 tests passed, 5.877 s |
| Combined new native-backend, IR, and cross-target suite | 50 tests passed, 16.049 s; no skips |
| Six generated demonstration streams | Every named output exactly matched the independent interpreter |
| New HEVC/AV1 hardware execution | Not tested |

The 50-test suite consists of 18 HEVC tests, 12 AV1 tests, 12 semantic/H.264
adapter tests, and eight cross-target tests. Times are test-suite durations,
not decoder-performance measurements.

```sh
python3 -m codec_compute.av1.bootstrap
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.demo

# Existing compiler baseline, from the repository root:
PYTHONPATH="$PWD:$PWD/05-h264-compiler" python3 -m unittest -v \
  test_compiler test_native_ops test_interpolation test_schedule
```

The historical H.264 baseline exercised available FFmpeg, OpenH264, and VA-API
decoder paths. New shared H.264 lowering was validated through FFmpeg. HEVC
uses FFmpeg's software decoder. AV1 native tests and demos run both libdav1d
and libaom-av1 through FFmpeg. NVDEC is unavailable on this host.

## Exactness and actual decoder work

- Averaging covers all 65,536 byte pairs on all three targets. The cross-target
  suite uses 16 chunks of 64x64 pixels to respect the existing H.264 writer's
  level-3.1 PCM access-unit bound. AV1 and HEVC native tests also use 256x256.
- HEVC saturating addition and subtraction each cover all 65,536 pairs in
  standalone instructions and resident programs. Thresholds cover every byte
  and every cutoff from -1 through 257. Signed weight and denominator boundaries
  exercise rounding before clipping.
- AV1 copies, positive and negative translations, compound averaging, repeated
  sources, odd sums, random images, and extrema agree with independent formulas
  through two decoders. All eight reference slots and repeated slot replacement
  are tested. Header inspection checks fixed entropy state, no inherited
  primary context, lossless quantization, and precise refresh masks.
- Changed input pixels retain identical computation tails. AV1 input-only
  serialization plus the neutral-compiled tail exactly matches a complete
  controlled encode with those same real inputs. Serializer spies verify that
  runtime receives inputs only. HEVC spies verify its sole upload call contains
  the original inputs and the encoder schedule's duplicate first input.
- Chains remain inside decoder reference buffers: HEVC tests retain an old
  input across 128 stages; AV1 tests retain and expose 20 composed averages.
  Outputs remain observable after reference slots are released.
- Shared tests replace the interpreter with a function that fails if called
  during encoding or decoding. Raw forged IR bounds fail compilation before
  they can incorrectly influence instruction selection.

## Demonstration artifacts

`python3 -m codec_compute.demo` generates the following at 32x32 logical pixels:

| Program | Target | Input pictures | Compute pictures | Prediction bytes |
| --- | --- | ---: | ---: | ---: |
| Two dependent rounded averages | H.264 | 2 | 2 | 343 |
| Two dependent rounded averages | HEVC | 2 | 2 | 34 |
| Two dependent rounded averages | AV1 | 2 | 2 | 60 |
| Saturating add → subtract → threshold | H.264 | 2 | 4 | 631 |
| Saturating add → subtract → threshold | HEVC | 2 | 4 | 71 |
| Average → clamp translation → average | AV1 | 2 | 3 | 92 |

These are serialization sizes, not speed measurements. H.264 includes guarded
I420 pictures; HEVC has different headers and geometry; AV1 uses monochrome IVF
and its byte total includes packet framing. Input-loading work and all emitted
bytes also matter when comparing complete invocations.

The two-average output SHA-256 is identical for all three codecs:

```text
4cefad337cdb8515e13a14b0de16f0a5f6b7ae67be98f8d73fb6e1dd1162be43
```

Generated streams, per-program manifests, and `validation.json` are available
under ignored `codec_compute/artifacts/` and can be regenerated by the demo.
The report records output hashes, prediction hashes, decoder names, and counts.

These results establish the implemented subset on the tested decoders. They do
not establish general AV1 affine/threshold synthesis, hardware portability,
cross-target runtime speedups, or a complete port of the original applications.
