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
denominators still produce zero. Chroma remains neutral. The default
`prediction_slice_rows=1` preserves row boundaries and the previous stream
bytes. Opt-in bands can join compatible rows, as described below.

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

## Opt-in prediction slice bands

Use `Writer(..., slice_mode="row", merge_zero_guards=True,
prediction_slice_rows=N)` or pass `h264_prediction_slice_rows=N` to shared
`compile` or `compile_graph`. The value must be a nonnegative integer; Boolean
values are rejected. Shared HEVC/AV1 compilation accepts only the default `1`.

- `1` is the byte-compatible default: one coded macroblock row per eligible
  prediction slice.
- A positive `N` groups at most `N` coded macroblock rows per slice, including
  guard rows. The last band may be shorter.
- `0` requests one slice for the whole coded prediction picture. Values above
  its coded row count have the same effect.

Banding requires enabled guard merging and the same uniform zero-offset
headers described above. Motion fields may vary within a band. Disabled guard
merging, biased predictions, differing block headers and macroblock mode retain
their previous slices. The option changes prediction slices only; PCM uploads
and parameter sets are unchanged.

For eligible predictions, the exact slice count is
`ceil(coded_macroblock_rows / effective_band_rows)`. The 16-pixel guard on each
side adds two coded rows to the active image's macroblock height:

| Active dimensions | Coded rows | `1` | `2` | `4` | `8` | `0` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128×128 | 10 | 10 | 5 | 3 | 2 | 1 |
| 256×256 | 18 | 18 | 9 | 5 | 3 | 1 |
| 512×512 | 34 | 34 | 17 | 9 | 5 | 1 |

Motion predictors reset at each slice boundary. Within a multirow slice, the
writer derives the H.264 16×16 predictor from available left, upper and
upper-right neighbors, with the specified upper-left fallback and matching
reference rules. Each reference list is handled independently. Thus joining
rows changes motion-vector differences as well as header count, while decoded
motion and pixels remain exact. Single-row slices retain their original
left-only encoding.

The manifest records the requested `prediction_slice_rows` and
`optimizations.prediction_slice_bands` with `requested_rows`,
`eligible_pictures` and `banded_pictures`. Use `costs.prediction_slices` and
`costs.prediction_bytes` for the actual emitted costs. Fewer slices do not
guarantee fewer bytes for every motion field, or lower execution time; this is
an explicit layout choice, not an automatic cost guard. Picture count, coded
surface size and output readback size do not change for a fixed prediction
plan. Existing shape, DPB, slice-rate and access-unit limits still apply, and
constructor bounds remain conservative enough for fallback operations.

Focused tests cover legacy bytes, exact slice counts, all fractional phases,
signed weights and denominators, fallback paths, malformed aliases, retained
references, multirow motion predictors and every decoded I420 byte through
available FFmpeg, OpenH264 and VA-API decoders. See the
[paired guard-merging experiment](../codec_compute/workloads/guard_fusion/README.md)
for measured CPU/decoder crossover and controlled background-work results.
