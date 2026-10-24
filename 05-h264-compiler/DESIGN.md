# Compiler v0.2

This is a symbolic image language embedded in Python. The kernel executes once
with symbolic images; static Python loops build a finite graph. Data-dependent
Python truth tests and comparisons raise errors. Floating-point operations,
dynamic shapes, and general PyTorch import
are outside this version.

## Language

Values are unsigned integers in [0,255]. The default input range [0,1] uses a
visible 0/255 seed normalized inside the decoder. Other ranges use literal byte
seeds, such as the maze's 0/128/255 input. Encoding validates every supplied pixel
and the static dimensions.

For independent inputs, pass `input_ranges={"a": (0,255), "b": (0,255)}` to
`compile()`. The compiler invokes the kernel with these keyword arguments.
Encoding, decoding, saving, and verification then take a dictionary of boards
with exactly those names. All inputs share the compiled dimensions. Named inputs
use literal bytes, including binary 0/1 inputs. The original single-input API
keeps its visible binary seed and normalization for compatibility.

| Operation | Exact meaning |
| --- | --- |
| `shift(x,dx=0,dy=0)` | Read `x[py+dy][px+dx]` at `(px,py)`; zero outside the board |
| `weighted(x,w=1,bias=0)` | `w*x+bias`, proven within [0,255] |
| `combine(a,b,wa=1,wb=1,bias=0)` | `wa*a+wb*b+bias`, proven within [0,255] |
| `add(a,b)` | Exact addition; alias for `combine(a,b)` |
| `saturating_weighted(x,w=1,bias=0)` | `clip8(w*x+bias)` |
| `saturating_combine(a,b,wa=1,wb=1,bias=0)` | `clip8(wa*a+wb*b+bias)` |
| `saturating_add(a,b)` | `min(255,a+b)` |
| `sample(x,qx=0,qy=0)` | H.264 interpolation at quarter-pixel offsets `(qx,qy)` |
| `gather(x,offsets)` | One `(qx,qy)` read offset per output 16×16 block, row-major |
| `threshold(x,t)` | 1 when `x>=t`, otherwise 0 |
| `bit_not`, `bit_or`, `bit_and` | Boolean operations on proven 0/1 operands |
| `equal(x,n)` | Exact equality built from thresholds and Boolean operations |
| `repeat(n,fn,x)` | Apply `fn` a static nonnegative integer number of times |
| `binary_dot(terms,weights,threshold)` | Signed binary dot product followed by a threshold |
| `conv2d_binary(x,weights,threshold)` | Centered 3×3 binary cross-correlation, zero padding |

`binary_dot` accepts 1–255 binary terms and matching ±1 weights. Convolution
accepts a nested 3×3 weight array. Both helpers expand into ordinary primitive
nodes; the backend has no neural-network-specific instructions.

Return one image or a mapping of names to images. Named outputs can include
generations, features, and masks; their original output frames remain available
in the decoded video without keeping those pictures live in the DPB. Return the
final result first and observations afterward for efficient dependency traversal.

Arithmetic has no implicit saturation or CPU fallback. `add()` on unrestricted
byte images is rejected because the range reaches 510; `saturating_add()` explicitly
requests the clipping behavior. Analysis is conservative:
even `combine(x,x,1,-1)` may fail without a proof accounting for correlations.
P coefficients fit [−128,127]; B combination coefficients fit [−64,63], since
they are doubled on the wire. The two B coefficients must additionally sum to
[−64,64] at denominator zero. The writer enforces the codec's hardware-weight
sum [−128,128], with upper limit 127 at denominator seven. A mathematically valid
u8 expression can still require a decomposition this frontend does not support;
such direct forms fail during compilation. Fusion preserves legal original
stages when their combined weights would violate this bound.

Saturating operations require a bias in [−128,127]. Splitting a larger offset
can change clipping, so the compiler rejects it. For example, unrestricted
`255-x` does not fit this one-picture API. Saturation applies after the complete
affine sum, including both inputs and bias. No algebra moves clipping across
operations.

## Spatial reads

Positive motion reads right/down, so visible content moves left/up. `sample()`
uses integer quarter-pixel coordinates: `qx=1` means +0.25 pixels. It uses H.264's
six-tap half-pixel filter, with quarter-pixel averages and specified rounding and
clipping. This is not bilinear sampling; edges can ring. The diagonal half-pixel
calculation preserves unrounded, unclipped intermediate sums. The independent
oracle checks all 16 phases, negative coordinates, and zero padding.

`gather()` supplies one motion vector per active 16×16 output block. A block can
copy any translated 16×16 region, including fractional positions; offsets are
relative to that output block. It supports tile permutations and piecewise
translations, not arbitrary per-pixel affine transforms. There must be exactly
`(width//16)*(height//16)` pairs. Both APIs use zero outside the active image.

## Graph and optimization

`graph.py` defines immutable nodes, range propagation, helpers, and iterative
traversal. `lower.py` visits reachable expressions, eliminates structurally
identical expressions (including commuted weighted pairs), removes Boolean
duplicates and double negation, and folds a single shifted or fractional read
into its consumer's motion vector. Exact affine chains fuse only when range
proofs exclude intermediate clipping and the final coefficients fit the codec.
Rounded operations and observable clipping remain separate.

Nested shifts cannot generally be added: `shift(shift(x,+1),-1)` loses its first
column at the intermediate boundary. The inner shift is materialized. Likewise,
`weighted(shift(x,1),bias=5)` gives 5 at the right edge, while
`shift(weighted(x,bias=5),1)` gives 0. Tests preserve both distinctions.

Nested fractional samples also materialize: two interpolations round and clip
twice and cannot be combined into one vector. A gather currently materializes
one P-picture; a following operation can reuse that reference.

`optimize=False` retains explicit shifts and repeated work, and selects the
legacy sliding reference window and one slice per macroblock. Some such
graphs exceed the reference budget even when their optimized forms fit; they
fail with a resource error instead of silently changing semantics.

### Semantic Boolean values

`ir.py` tracks complete scalar truth signatures over at most two independent
binary frontier values. Each table has at most four entries, including exact
nonbinary arithmetic intermediates and explicit clipping. It removes irrelevant
variables, aliases projections to their inputs, shares equivalent Boolean
expressions, and replaces constant Boolean results with one legal prediction.
Constant generation remains real decoder work unless an existing value already
provides the constant.

Facts propagate bottom-up, so a long recurrence can collapse without increasing
the bounded instruction search's cut size. For example, XOR with the same B
twice aliases A; forty repetitions do too. More than two distinct frontiers
stop propagation. Binary results beyond that boundary become opaque values.
Spatial operations also open opaque frontiers, preserving crop and zero-padding
boundaries. Singleton ranges can still prove spatial zeros constant.

This pass preserves named outputs, including intermediates. It never evaluates
actual input pixels. A whole-graph scalar instruction comparison rejects
rewrites that would cost more after later fusion/search. If the old lowering
cannot represent a dead expression, a legal simplified result may still compile;
its saved-instruction count is reported as unknown. This comparison does not
claim globally optimal reference allocation, packed pictures, or latency.

`compile(..., semantic_ir=False)` disables semantic value propagation while
keeping existing optimizations and exact search enabled. `optimize=False`
disables it too. Manifest `semantic_ir_*` statistics record accepted rewrites,
savings, and truth signatures. Their value IDs belong to the semantic pass,
not physical frames or final graph IDs. These records explain local decisions;
they are not standalone proof certificates containing the full original DAG.

`Program.reference()` retains and interprets the original traced graph, before
any IR rewrite. `verify()` therefore checks decoded results against those
original semantics. The same independent oracle is retained by packed programs.

### Bounded exact instruction search

`search.py` discovers scalar subgraphs with binary outputs and one or two input
values. It considers at most 24 cuts per graph node, 32 internal nodes per cut,
and 64 combinations in the complete Cartesian input domain. Movement operations
form boundaries. Named results and intermediates used outside a cut cannot
disappear. Repeated internal uses are allowed, so reconvergent predicates such
as Life's survival/birth expression can be optimized together.

`synthesize.py` searches a finite grammar of one, two, or three exact weighted
predictions. Single predictions explore every signed P weight and a documented
subset of B weights. Longer candidates use small integer projections, clipping,
and rounded combinations of one or two clipped projections. Integer constraints
fit legal offsets and final weights. Every candidate is then checked against
every input combination using the codec's signed arithmetic, rounding, and
saturation. Results and failures share a bounded 512-entry cache. This proves
equivalence within declared ranges; it does not prove global optimality or
that an unsupported function has no H.264 implementation.

A replacement must beat the existing lowering for its cut. A final whole-graph
comparison also rejects changes that fail to reduce instruction count, including
lost fusion opportunities at cut boundaries. The original graph remains intact
for independent verification. `compile(..., exact_search=False)` disables only
this pass; `optimize=False` also disables it. No actual input pixels are needed
for search, and the resulting prediction tail remains reusable.

Manifest statistics include `exact_search_rewrites`,
`exact_search_saved_instructions`, and `exact_search_proofs`. Each proof records
the root and frontier graph IDs, inclusive ranges, complete truth table, old
instruction count, and replacement steps. Step inputs index frontier values
first, then preceding step results; weights are actual codec weights, and bias
is applied to both source offsets for B prediction. Physical frame assignments
remain in the ordinary instruction manifest.

## Exact lowering

With denominator zero, weighted P prediction computes `clip8(w*x+offset)`.
Explicit weighted B prediction computes:

```text
clip8(((w0*a+w1*b+1)>>1) + ((o0+o1+1)>>1))
```

Weights `(2*wa,2*wb)` and equal offsets `(bias,bias)` therefore implement exact
`clip8(wa*a+wb*b+bias)`. For strict operations, range proofs make clipping inactive. Large affine biases
are split into legal offsets only if every intermediate also fits.

Threshold selection first searches legal P weights, denominators, and offsets.
A candidate must give 0 below the cutoff, 1 at it, and at most 1 at the largest
possible input. Monotonicity proves the intervening values. Negative results on
the false side may clip safely to zero.

Binary OR lowers directly to one B prediction: weights `(1,1)`, denominator
zero, and zero offsets compute `(a+b+1)>>1`. AND uses weights `(2,2)` and
offsets `(-1,-1)` to compute `clip8(a+b-1)`. A threshold over an unnamed,
single-use sum of binary leaves becomes a chain of these gates when its cutoff
means OR or AND. Shared or named arithmetic sums keep their original values.

If a one-picture threshold is unavailable, the compiler can use
`clip8(1-clip8(t-x))`. This takes two P pictures with negative weights and is
selected only when its offsets are legal and it saves a picture.

When one prediction is insufficient, any byte comparison uses:

```text
clip8(((clip8(x + 192 - t) + 64) >> 7) - 1)
```

This yields exactly 0 below threshold `t` and 1 at or above it. Shift offsets
are split into legal parts with the same sign, so saturation already lies on
the correct side. Cutoffs outside the proven range become constant results.
Tests exhaust every byte and cutoffs −1 through 257 on all three decoders.

Binary dot products use bias `N`, the negative-weight count. Both
`N+sum(w*x)` and its partial sums stay within [0,number_of_terms]. The ordinary
threshold then compares this nonnegative score with `N+threshold`, handling
small convolutions and 196-input classifier neurons using the same lowering.

## Scheduling, output, and verification

`schedule.py` computes last uses and assigns live pictures to explicit long-term
reference slots. Reference-list modification selects each slot; memory-management
commands replace a dead occupant after the current prediction finishes. A picture
may remain available for more than 16 decoding steps, eliminating age-driven
refresh copies. The 16-picture reference capacity is unchanged.
At most 15 simultaneously live values are supported before an instruction.
More demanding graphs fail explicitly; this version has no spill or
recomputation strategy. Iterative graph traversal supports thousands of steps.

`backend.py` specializes dimensions, calls the existing shared bitstream writer,
caches the prediction tail, and writes the stream and manifest. `save(path,image)`
requires the initial pixel state, and rejects a path colliding with its sibling
JSON manifest. Dimensions must be multiples of 16 and satisfy the writer's
High-profile level-3.1 limits. Motion offsets are limited here to ±127 pixels.
Every picture retains a 16-pixel zero guard and neutral chroma; deblocking is off.
Prediction slices group consecutive macroblocks within a row that share weights
and references. Their motion-vector differences account for the preceding
macroblock; guard and active regions split when their prediction parameters differ.
For the motion demo, this reduces 48 prediction slices to 14 per picture.

For controlled comparisons, `compile(..., reference_mode="sliding")` restores
age-based copies, and `slice_mode="macroblock"` restores independent slices.
The optimized defaults are `"retained"` and `"row"` respectively.

Input zero is an IDR I_PCM picture. Further inputs are non-IDR I_PCM references,
preserving the pictures already loaded. Computation starts at frame `input_count`.
Up to 15 input pictures can be loaded, subject to the same live-value budget;
two sources can be combined per instruction. `encode_inputs()` exposes just this
loading step for measurement. `program.tail` contains only reusable predictions.
Host input bytes must still be serialized and decoded on every new invocation.
This does not inject existing GPU tensors into the DPB.

The shared `decoder_core.decode.decode_batch()` runtime can execute a finite
list of independent same-shape streams in one decoder session. Each job begins
with its own SPS/PPS and IDR and must have the declared picture count. Dimensions
refer to full coded pictures, including guards. `selected_frames=(...)` gives
strictly increasing picture indices within each job; results are packed I420
bytes indexed by job and position in that selection. On NVDEC, only selected
pictures undergo VIC/CPU readback. Other output samples are released while the
decoder retains its internal prediction references. Compatibility backends
still read every picture before selecting on the host. This API batches existing
streams through EOS; it does not maintain an interactive service or reuse input
references across job boundaries. See [motion-mask timings](../06-motion-mask-decoder/VALIDATION.md).

`Program.decode(..., readback="outputs")` and `verify(..., readback="outputs")`
use this runtime for a single invocation and return every named output. Aliases
share one physical readback; `output_picture_indices` is the sorted unique tuple.
`Program.read(raw, selected_frames=indices)` validates strictly increasing,
in-range integer indices, exact packed size, complete named-output coverage,
each supplied picture's guard/chroma, and each named output's proven range.
Missing intermediates are never filled with invented pixels. Full readback
remains the default for the shared API and checks guard/chroma in every picture.
Selective readback checks those properties only in supplied pictures; use full
validation when auditing a new program. The maze CLI defaults to output selection
after both modes have been independently checked against full decoder output.

The manifest records logical graph IDs/ranges, physical instruction frame numbers,
reference dependencies, weights/offsets/motion, output frames, refreshes, and
optimization statistics. It also records slot assignments, execution modes and
`costs`: input, normalization, arithmetic and refresh pictures, prediction slices
and bytes, and peak live values. `program.costs` exposes the same information.
Folded nodes and aliases may have no standalone frame.
The prediction hash identifies reusable instructions; the stream hash also
includes the particular seed.

Manifest format `h264-python-0.2` adds named input ranges, encodings, and physical
frames. Instructions include integer and quarter-pixel motion components and
optional block offset tables. `operations` remains a prediction-only list;
enumerate it starting at `program.input_count` to obtain physical frame numbers.

`verify.py` interprets graph semantics independently of lowering. `verify()`
compares every named output pixel with actual decoding. Extraction checks picture
count, zero guards, neutral chroma, and proven output ranges. No reference
evaluation of supplied images occurs in `compile()`, `encode()`, `save()`, or
`decode()`. Compilation may enumerate small scalar truth tables to prove rewrites.

Binary computational outputs contain 0/1; previews scale them afterward. Maze
paths are extracted from decoded arrival times. MNIST reads one decoded class
bit per tile; input binarization and class-name formatting are host preprocessing
and presentation. Compilation uses only the Python standard library. Optional
checkpoint comparisons require the original PyTorch environment.
