# 09 — Exact image IR for decoder computation

Experiment 09 is the current frontend for new decoder-compute programs. It
separates exact image semantics from target instruction selection, scheduling
and execution, with bounded H.264, HEVC and AV1 backends. Start here for
navigation and migration from [experiment 05](../05-h264-compiler/README.md).

The Python package remains [`codec_compute`](../codec_compute/README.md) at the
repository root. Imports, source paths, tests and saved artifacts have not
moved into this numbered directory. Experiment 05's frontend is deprecated for
new work, but its backend components and packed workloads remain in use.

## Start with an exact kernel

Run from the repository root with Python and FFmpeg available:

```python
import codec_compute as media

def blend(a, b):
    return {"output": media.average(a, b)}

program = media.compile(
    blend, target="h264", width=32, height=32,
    input_ranges={"a": (0, 255), "b": (0, 255)},
)
inputs = {"a": [[10] * 32 for _ in range(32)],
          "b": [[21] * 32 for _ in range(32)]}
assert program.verify(inputs, decoder="ffmpeg")["output"][0][0] == 16
program.save("blend.h264", inputs)
```

`average` rounds the full sum upward at a half: `(a+b+1)//2`. `encode` uploads
inputs and appends a cached computation tail; `decode` returns actual decoder
outputs. `reference` interprets the semantic graph, and `verify` compares both.
No host-computed output is encoded as the answer. Use `trace` followed by
`compile_graph` to lower the same semantic graph for different targets.

Target support differs. H.264 implements the broadest current spatial subset;
HEVC supports weighted arithmetic and bounded scalar synthesis; AV1 supports
copy, clamp-border integer translation and rounded averaging, plus successful
synthesis from those primitives. Unsupported operations fail explicitly. See
the [capability matrix](../codec_compute/README.md#current-capability-matrix)
before changing `target`; AV1 also requires its explicitly invoked serializer
[bootstrap](../codec_compute/av1/README.md). New HEVC/AV1 hardware execution is
not established by the H.264/NVDEC measurements.

## Navigation

| Topic | Implementation and evidence |
| --- | --- |
| Public semantics, target limits and local execution | [API reference](../codec_compute/README.md), [validation](../codec_compute/VALIDATION.md) |
| Exact scalar instruction discovery | [Bounded synthesis](../codec_compute/synthesis/README.md) |
| Change detection, extrema, masks and smoothing | [Workload survey](../codec_compute/workloads/README.md) |
| Input serialization versus pre-encoded decoding | [Temporal benchmark](../codec_compute/workloads/temporal/README.md) |
| Shift fusion and named-output preservation | [Spatial fusion](../codec_compute/workloads/motion-fusion/README.md) |
| Supplied-motion interpolation, routing and residuals | [Resampling experiments](../codec_compute/workloads/resampling/README.md) |
| Repeated jobs through one decoder pipeline | [Resident runtime](../codec_compute/workloads/resident/README.md) |
| Native CPU baselines and larger inputs | [CPU comparison](../codec_compute/workloads/cpu_comparison/README.md) |
| Fewer prediction slices and controlled CPU contention | [Guard merging](../codec_compute/workloads/guard_fusion/README.md) |
| Retained legacy packing | [Experiment 05 packing](../05-h264-compiler/README.md#packing-independent-predictions), [experiment 07](../07-packed-denoising-decoder/README.md) |

These are implementation comparisons with exact output checks. Decoder
execution does not imply a speedup. The current CPU and guard-merging reports
show CPU winning elapsed time at every tested depth; some longer synthetic
chains save measured process CPU. The small observed background-work gain has
overlapping trial ranges and does not establish general application benefit.

## Migrate from the experiment 05 frontend

This is a source-level migration, not an import alias. The two packages have
different graph and program types; rebuild kernels using `codec_compute`
operations rather than passing legacy objects into `compile_graph`.

| Experiment 05 | Experiment 09 |
| --- | --- |
| `import h264` with `05-h264-compiler` on `PYTHONPATH` | `import codec_compute as media` from the repository root |
| Optional `@h264.kernel` marker | Ordinary Python function; no decorator required |
| `h264.compile(fn, input_range=(0, 1))` | `media.compile(fn, input_ranges={"image": (0, 1)})`; the function must accept keyword `image` |
| Omitted input range defaults to one binary input | Omitted `input_ranges` defaults to one full-byte input `(0, 255)` |
| A single unnamed input can be supplied as a board | Always supply a mapping, such as `{"image": board}` |
| `program.verify(board, backend="ffmpeg")` | `program.verify({"image": board}, decoder="ffmpeg")` |
| `program.costs`, `program.pictures` | `program.manifest()["costs"]`, including `total_pictures` |
| `h264-python-0.2` manifest and legacy graph records | `codec-compute-program-1` manifest containing semantic `ir`, `lowered_ir`, synthesis and target records |

With explicit `input_ranges`, names become kernel keyword arguments in both
frontends. Without it, the new frontend invokes one positional argument but
still names the supplied pixel board `image`. Boolean operators require proven
0/1 inputs, so specify their ranges when migrating binary kernels. The old
single-input binary path uploads a visible 0/255 seed and normalizes it; the
new H.264 adapter uploads literal 0/1 bytes. Logical results can match while
input pictures, intermediate pictures and stream bytes differ. Named binary
inputs in the old API already use literal samples.

Several operations also change names or coordinate units:

| Legacy operation | Shared IR equivalent |
| --- | --- |
| `weighted(x, w, bias)` | `affine(x, weights=(w,), bias=bias)` |
| `combine(a, b, wa, wb, bias)` | `affine(a, b, weights=(wa, wb), bias=bias)` |
| `saturating_weighted` / `saturating_combine` | Corresponding `affine(..., saturating=True)` |
| `saturating_add(a, b)` | `add_sat(a, b)` |
| `sample(x, qx, qy)` | `sample_h264(x, qx, qy)`; quarter-pixel units and H.264 filtering |
| `gather(x, offsets)` | `gather_h264(x, offsets, block_size=16)`; preserves the old quarter-pixel units |
| `shift(x, dx, dy)` | `shift(x, dx, dy, border="zero")`; integer-pixel units |
| `threshold(x, t=...)` | `threshold(x, cutoff=...)`; the predicate remains `x >= cutoff` |
| `repeat(n, fn, value)` | Static Python `for` loop that rebuilds the symbolic value |

**Do not rename legacy `gather` to `media.gather` unchanged:** the latter takes
integer-pixel offsets. Use `gather_h264` for the original meaning. Fractional
sampling remains codec-qualified, with zero extension, six-tap filtering and
exact clipping/rounding; it is not portable bilinear interpolation.

`add`, `threshold`, `bit_not`, `bit_and` and `bit_or` retain their image
meanings. The new API additionally exposes rounded `average`, `sub_sat`,
`minimum`, `maximum`, `absdiff` and `bit_xor`. Boolean operations act on 0/1
images, not the bits of arbitrary bytes. Strict affine operations reject
possible u8 overflow; saturation must be explicit. Target lowering can still
reject an otherwise valid semantic operation.

## What has not migrated

There is no shared `pack` or `PackedProgram` API. `h264.pack` accepts an
unpacked legacy `h264.Program`; neither a shared `Program` nor its native
adapter is a compatible replacement. Continue using experiment 05 for existing
packing, mixed P/B grouping and stable/critical/auto packed scheduling. Spatial
block gathering in the shared IR is not independent-operation packing and does
not shrink the coded output surface.

The shared frontend has no public `Input`, `equal`, `binary_dot` or
`conv2d_binary` helper. Define inputs through tracing; port helper formulas
explicitly and verify intermediate range/clipping behavior, or retain the
legacy workload. Equality can be expressed with threshold/Boolean operations,
but there is no automatic helper or saved-graph converter.

Legacy compile controls `optimize`, `exact_search`, `semantic_ir`,
`reference_mode` and `slice_mode` are not shared compile keywords. Shared
`synthesis` and its search budgets control a different bounded pass; they do
not reproduce the entire old optimizer. H.264-specific controls are
`h264_fuse_shifts`, `h264_level` and `h264_merge_zero_guards`. The shared target
uses retained references and row slices, with compatible guard merging enabled
by default. Disabling that pass alone does not promise old frontend bitstreams
or instruction counts.

The shared `Program` does not expose the legacy `read(raw, selected_frames=...)`
or `decode(..., readback="outputs")` interface. Its H.264/NVDEC `session()` API
supports selected output readback through `session.run(inputs)`, with explicit
completion fences and fixed-shape jobs. This is a different runtime contract,
not a replacement for all old batch/reader workflows.

Experiment 05 is therefore **not retired**. The shared H.264 adapter still loads
its scalar selector, retained-reference scheduler and I420 reader from
`05-h264-compiler/h264`; both use `decoder_core` for bitstream and decoder tools.
Keep those directories together. Existing numbered experiments, training
weights, captures and historical reports remain at their original paths.
