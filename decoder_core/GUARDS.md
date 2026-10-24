# Merging zero guards into prediction rows

`Writer(..., slice_mode="row", merge_zero_guards=True)` can reuse a compatible
row's prediction header for its zero-valued border. The ordinary Writer default
is `False`, preserving existing byte streams. The shared IR compiler enables
this option by default; pass `h264_merge_zero_guards=False` to `compile` or
`compile_graph` for the baseline.

Eligibility is deliberately narrow: row slices, every source offset exactly
zero, and identical reference/weight/offset/denominator headers across the
parent and all block overrides. Motion may vary between active blocks. Guard
blocks keep the same header but read their own coordinates with zero motion.
Because their reference samples are zero, all supported signed weights and
denominators still produce zero. Chroma remains neutral. Motion predictors are
reset at row boundaries exactly as before; rows are never concatenated.

The contract requires zero-guard reference pictures, as established by Writer
PCM uploads and maintained by every computation picture. A neutral-128 runtime
completion fence must not be used as a computation reference. Existing sessions
reset at the next input IDR and never reference such a fence.

Nonzero offsets, differing block headers, and macroblock slice mode use the
previous guard policy. Even offsets that cancel under B prediction fall back.
All original source and motion validations still run. For a fixed prediction
plan, input serialization, reference allocation, prediction arithmetic, frame
count and decoded outputs are unchanged; eligible prediction bytes and slice counts change.

| Active dimensions | Previous row slices/picture | Merged slices/picture |
| --- | ---: | ---: |
| 128×128 | 26 | 10 |
| 256×256 | 50 | 18 |
| 512×512 | 98 | 34 |

All-zero-weight operations already share row headers and need no additional
merging. The shared manifest records `merge_zero_guards` and
`optimizations.zero_guard_merge` with enabled/eligible/merged picture counts.
Existing resource bounds remain conservative, including constructor checks
that allow for unmerged operations.

Focused tests cover legacy bytes, exact slice counts, all fractional phases,
signed weights and denominators, fallback paths, malformed aliases, retained
references and every decoded I420 byte through available FFmpeg, OpenH264 and
VA-API decoders. See the [paired experiment](../codec_compute/workloads/guard_fusion/README.md)
for measured CPU/decoder crossover and controlled background-work results.
