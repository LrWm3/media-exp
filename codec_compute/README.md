# Exact image IR targeting H.264, HEVC, and AV1

This experiment follows **preserved H.264 → independent minimal HEVC → shared
compute IR → AV1 → revised IR**. All three targets emit real video bitstreams;
unmodified decoders execute their predictions. The implementation deliberately
supports a bounded, verified subset. Existing H.264 applications remain in
`05-h264-compiler` without modification.

```python
import codec_compute as media

def kernel(a, b):
    mean = media.average(a, b)
    return {"output": media.average(mean, a), "mean": mean}

graph = media.trace(kernel, width=32, height=32,
                    input_ranges={"a": (0, 255), "b": (0, 255)})
inputs = {"a": [[10] * 32 for _ in range(32)],
          "b": [[21] * 32 for _ in range(32)]}
for target in ("h264", "hevc", "av1"):
    program = media.compile_graph(graph, target=target)
    assert program.verify(inputs)["output"][0][0] == 13
```

`compile(kernel, target=..., width=..., height=..., input_ranges=...)` combines
tracing and lowering. An omitted `input_ranges` declares one positional input,
called `image` when supplying pixels. Named inputs invoke the kernel by keyword.

The compiler also performs [automatic scalar synthesis](synthesis/README.md)
for `minimum`, `maximum`, `absdiff`, and Boolean AND/OR/XOR. It searches each target's exact
primitive library, exhaustively verifies candidates, and lowers successful
plans into decoder instructions. H.264/HEVC find all six goals; the initial
AV1 library finds OR and reports bounded search misses for the others.

[Useful workload survey](workloads/README.md) exercises change detection,
temporal extrema, blending, binary dilation, and explicitly rounded smoothing.
Fourteen task/target combinations execute exactly. Absolute difference needs
three computation pictures. H.264 spatial read fusion halves the dilation and
smoothing picture counts. Captures, input-loading sizes, target
restrictions, and exact-decoder evidence are included.
The [temporal benchmark](workloads/temporal/README.md) separately measures input
serialization, pre-encoded decoding, and complete host-input-to-output runs.

## Run locally

Run commands from the repository root. Python uses its standard library.
H.264 needs FFmpeg; HEVC additionally needs its `libx265` encoder and
`trace_headers` bitstream filter. AV1 needs an explicit one-time serializer build:

```sh
python3 -m codec_compute.av1.bootstrap
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.demo
python3 -m codec_compute.synthesis.demo
```

The bootstrap downloads a pinned libaom revision and builds a small, narrowly
patched serializer outside this checkout. It requires Git, CMake, and a C/C++
compiler. Details and source pin: [AV1 backend](av1/README.md). Compilation never
downloads or installs dependencies automatically. AV1 native tests skip when
the build is missing; a complete three-target validation requires bootstrap.

The demo writes six verified `.h264`, `.hevc`, or `.ivf` streams, JSON manifests,
and `validation.json` under ignored `codec_compute/artifacts`. Use
`--target h264` or `--target hevc` before installing the AV1 serializer.

## Semantics and implementation boundary

```text
Python kernel → exact u8 image graph → bounded synthesis + scalar verification
                                    → target-specific instruction selection
                                    → codec-specific reference scheduling
                                    → lossless input prefix + fixed compute tail
                                    → ordinary decoder
```

`ir.py` defines immutable image values, interval proofs, named inputs/outputs,
and an independent interpreter. It checks user-constructed nodes' topology and
recomputes their range proofs before any target can trust them. It contains no
P/B frame types, macroblock sizes, coefficient limits, reference slots, or
entropy-coding syntax. Target resource restrictions are checked during lowering.

Exact affine arithmetic rejects possible overflow. `add_sat`, `sub_sat`, and
`affine(..., saturating=True)` clip only after the complete affine sum.
`average(a,b)` computes **`floor((a+b+1)/2)`**, retaining the full sum before
rounding. `shift(x,dx,dy,border=...)` reads `x[y+dy,x+dx]`; borders are explicitly
`zero` or `clamp`. Boolean operations require proven 0/1 inputs.

`encode()` validates and serializes inputs, then appends the cached instruction
bytes. `decode()` extracts actual decoder results. Only `reference()` and
`verify()` invoke the mathematical interpreter. Computation tails never depend
on actual input pixels. Multiple stages keep intermediate values in decoder
reference buffers; no intermediate is read back and re-encoded for execution.

The H.264 adapter folds eligible zero-border shifts into scalar consumers, then
reuses the selector, retained-reference scheduler, writer, and reader. Rounded
average lowers directly to a weighted B prediction. Named shifts and successive
crop boundaries remain materialized. Both fused and unfused plans are scheduled;
fusion is accepted only when computation pictures, prediction bytes, and distinct
output readback pictures do not increase. The manifest records that decision.
Pass `h264_fuse_shifts=False` to `compile` or `compile_graph` for the baseline;
this target-specific option has no effect on HEVC or AV1. See the
[spatial comparison](workloads/motion-fusion/README.md).

The HEVC backend was implemented independently using lossless libx265 uploads,
verified skip-CABAC templates, and compiler-written weighted-prediction headers.
The AV1 backend uses a pinned serializer to select references and integer motion,
force zero residuals, and emit rounded compound prediction. At runtime its
serializer receives input frames only. Neither backend encodes host-computed
results. Their native APIs remain available for codec-specific experiments:
[HEVC](hevc/README.md), [AV1](av1/README.md).

## What AV1 changed in IR revision 2

1. **Rounded average remains a first-class operation.** A decomposition into
   u8 addition followed by division would reject or clip valid inputs. For
   example, averaging 255 with 255 must return 255, not 128.
2. **Border behavior is semantic.** AV1's implemented motion repeats edge
   samples; the H.264 writer uses zero guards. The adapter rejects an unsupported
   border instead of changing the requested computation.
3. **Reference and output identities are separate.** A value's named output is
   observable after its reference slot is released. Each target owns slot
   allocation and decoded output ordering. HEVC exposes distinct POCs and
   displayed-frame indices; AV1 refreshes eight physical slots.
4. **Unsupported means unimplemented unless proved otherwise.** AV1 rejects
   general affine and threshold nodes. That says nothing about whether a more
   elaborate sequence could implement them. No approximate or CPU fallback runs.

This IR is not the intersection of target capabilities: signed affine arithmetic,
thresholds, and both border modes remain expressible even when one target cannot
lower them. Codec-defined fractional sampling is intentionally not given a
portable meaning; the original H.264 API remains available for those operations.

## Current capability matrix

`Native` means a direct prediction; `sequence` means several verified predictions.
`Unimplemented` describes this backend, not a codec impossibility result.

| Computation | H.264 | HEVC | AV1 |
| --- | --- | --- | --- |
| Copy | Native or alias | Native | Native |
| Rounded average | Native | Native | Native |
| Saturated add/subtract | Native, legal coefficients | Native, legal coefficients | Unimplemented |
| Signed affine | Native or supported offset sequence | Native, legal coefficients/offset | Unimplemented except identity |
| Threshold | Native or sequence | Sequence | Unimplemented |
| Minimum / maximum | Synthesized, 2 predictions | Synthesized, 2 predictions | No plan within default search bound |
| Absolute difference | Synthesized, 3 predictions | Synthesized, 3 predictions | No plan within default search bound |
| Boolean OR | Synthesized rounded average | Synthesized rounded average | Synthesized rounded average |
| Boolean AND / XOR | Synthesized, 2 / 3 predictions | Synthesized, 2 / 3 predictions | No plan within default search bound |
| Boolean NOT | Native | Native | Unimplemented |
| Integer shift, zero border | Native | Unimplemented | Unimplemented |
| Integer shift, clamp border | Unimplemented | Unimplemented | Native, each component ±16 |

| Implementation constraint | H.264 adapter | HEVC adapter | AV1 adapter |
| --- | --- | --- | --- |
| Uploaded inputs | Up to 15 | 1–2 | 1–8 |
| Live reference policy | Existing 15-live-value scheduler | At most 3 retained references | 8 slots, including slot needed for next result |
| Shape | Multiples of 16; existing level and PCM limits | Multiples of 8, 16–512 each axis | Multiples of 16, 16–4096 each axis |
| Instruction bound | Existing writer/reference constraints | 128 native prediction stages | No fixed bound in allocator |
| Validated execution | FFmpeg, OpenH264, VA-API; workload suite also NVDEC | FFmpeg HEVC software decoder | FFmpeg/libdav1d and FFmpeg/libaom-av1 |

Manifests report input/computation pictures, prediction bytes, outputs, and
target scheduling information. They are static work counts, not runtime speed
claims. In particular, HEVC's binary uploader encodes a duplicate input for its
temporary encoder schedule and discards that duplicate's slice; two uploaded
pictures remain in the final stream. Neither compression efficiency nor newer
codec age establishes faster computation. New HEVC/AV1 hardware execution remains
unvalidated here.

## Validation and next research questions

Tests cover every u8 pair for averaging across all three targets; signed and
clipped arithmetic; intermediate clipping; exact Boolean OR; changed inputs with
identical compute bytes; no interpreter invocation during encoding/decoding;
output aliases; saved manifests; resource rejection; and per-target borders.
Native tests additionally inspect AV1 entropy/reference headers, exercise full
reference capacity and reuse, and verify long HEVC chains. See
[VALIDATION.md](VALIDATION.md) for the recorded run and reproducible commands.

The next useful extensions are richer synthesis primitive catalogs,
target-aware costs, broader reference allocation, controlled hardware validation,
and codec-specific operations beyond this initial set. MLIR is not required for
the current graph size or pass structure.
