# Exact image IR targeting H.264, HEVC, and AV1

This is the implementation and API reference for
[experiment 09](../09-codec-compute-ir/README.md), the current frontend for new
decoder-compute programs. Its numbered landing page collects workload navigation
and the migration guide from experiment 05. The Python package and artifacts
remain at their existing `codec_compute` paths.

All three targets emit real video bitstreams; unmodified decoders execute their
predictions. The implementation supports a bounded, verified subset. Existing
`h264` applications and packing remain in `05-h264-compiler`; the shared H.264
adapter still depends on that directory's selector, scheduler and reader.

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
The [resampling experiments](workloads/resampling/README.md) explore patch atlases,
supplied-motion interpolation, bandwise correction, compensated residuals,
dense polarity maps, and stereo cost candidates.
The [resident runtime experiment](workloads/resident/README.md) adds explicit
High level 4.1 for 256×256 input uploads and sequential NVDEC requests through
one live pipeline, with tiny counted IDR fences and exact output validation.
The [CPU comparison](workloads/cpu_comparison/README.md) tests optimized CPU
kernels against warm decoding, native input serialization, retained intermediate
chains and finite batches, with separate elapsed and process-CPU measurements.

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
this compatibility flag now controls all spatial read fusion and has no effect
on HEVC or AV1. See the
[spatial comparison](workloads/motion-fusion/README.md).

## Block routing and explicit H.264 sampling

For larger uploads, pass `h264_level=41` or `h264_level=51` to `compile` or
`compile_graph`. The writer retains level `31` by default. All
three supported levels have checked resource limits; levels 4.1 and 5.1 also
use conservative per-AU bitrate caps. These enable the tested 256×256 and
512×512 byte inputs, respectively, without promising arbitrary image sizes. See [level constraints](../decoder_core/LEVELS.md).

The shared compiler also enables zero-guard row merging by default. Compatible
zero-offset predictions reuse their headers across the border, reducing the
512×512 row layout from 98 to 34 slices without changing any decoded pixels.
Pass `h264_merge_zero_guards=False` to recover the baseline layout. The direct
Writer defaults to disabled, preserving its previous stream bytes. Manifests
record eligibility and actual merged-picture counts; biased or heterogeneous
headers fall back. See [guard semantics](../decoder_core/GUARDS.md) and the
[paired performance experiment](workloads/guard_fusion/README.md).

`with program.session(helper="/path/to/nvdec_decode") as session:` keeps an
NVDEC process and pipeline alive; `session.run(inputs)` returns named image
outputs after each independent job. It serializes new inputs, reuses cached
instruction bytes, and checks selected I420 guards, chroma and output ranges.
A tiny extra IDR completes each request; `session.stats` counts those fence
pictures and bytes. Close the context to validate delayed diagnostics and drain
the final fence. Job references reset at IDR boundaries. This runtime is H.264/
NVDEC only, with fixed shape and compile-time motion fields; it does not reuse
camera frames or GPU reference arrays across jobs.

`gather(image, offsets, block_size=16, border="zero")` uses an immutable table
of integer-pixel `(dx,dy)` offsets, one per output block in row-major order.
Each pixel reads its original source at that relative position. The semantic
oracle supports zero/clamp borders and partial edge blocks; the H.264 adapter
currently requires 16×16 blocks, zero borders, and its existing aligned shape.
The output retains the graph's shape, so packing a smaller ROI does not shrink
the coded picture or hardware readback surface.

`sample_h264(image, qx, qy)` and `gather_h264(image, offsets, block_size=16)`
explicitly name the [H.264 luma interpolation rules](https://www.itu.int/rec/T-REC-H.264-202408-S/en)
(8.4.2.2.1). Their offsets are integer **quarter-pixel** units. They extend the
active image with zero before six-tap filtering, retain the specified diagonal
intermediate precision, and apply codec-defined rounding and clipping. Fractional
samples may overshoot an input's narrower range, so their proof is `[0,255]`.
They are not generic bilinear resampling operations. Other targets currently
reject these qualified operations.

```python
def halfway(a, b):
    # Supplied motion, ±2.5 pixels; no optical-flow estimation in the decoder.
    return media.average(media.sample_h264(a, 10, 0),
                         media.sample_h264(b, -10, 0))

program = media.compile(halfway, width=128, height=128,
                        input_ranges={"a": (0, 255), "b": (0, 255)})
assert program.manifest()["costs"]["computation_pictures"] == 1
```

Two independent per-block fields can likewise feed one rounded-average B
prediction. The compiler schedules their parent references, then applies those
physical frame identities to every block; each block has its own motion vectors.
Gather or sampling results can also fold into affine arithmetic and threshold
sequences. Nested motion/filter stages remain separate, as do named sampled
outputs. Motion follows the shared writer limits: horizontal quarter-pixel
components lie in `[-8192, 8191]`, vertical components in `[-2048, 2047]`.
Fields are compile-time constants: changing them requires compiling a new prediction tail.
Manifests include `spatial_read_fusion` decisions and prediction slice counts.

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
lower them. Codec-defined fractional sampling uses explicitly qualified
`sample_h264` and `gather_h264` operations; it does not acquire a portable
bilinear meaning. The original H.264 API remains available too.

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
| Integer block gather, zero border | Native, 16×16 blocks | Unimplemented | Unimplemented |
| Explicit H.264 fractional sample / block gather | Native, can fuse into scalar reads | Unimplemented | Unimplemented |

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
and additional codec-specific operations. MLIR is not required for
the current graph size or pass structure.
