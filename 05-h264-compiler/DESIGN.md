# Compiler v0.1

This is a symbolic image language embedded in Python. The kernel executes once
with a symbolic image; static Python loops build a finite graph. Data-dependent
Python truth tests and comparisons raise errors. Floating-point operations,
dynamic shapes, multiple independent input pictures, and general PyTorch import
are outside this version.

## Language

Values are unsigned integers in [0,255]. The default input range [0,1] uses a
visible 0/255 seed normalized inside the decoder. Other ranges use literal byte
seeds, such as the maze's 0/128/255 input. Encoding validates every supplied pixel
and the static dimensions.

| Operation | Exact meaning |
| --- | --- |
| `shift(x,dx=0,dy=0)` | Read `x[py+dy][px+dx]` at `(px,py)`; zero outside the board |
| `weighted(x,w=1,bias=0)` | `w*x+bias`, proven within [0,255] |
| `combine(a,b,wa=1,wb=1,bias=0)` | `wa*a+wb*b+bias`, proven within [0,255] |
| `add(a,b)` | Exact addition; alias for `combine(a,b)` |
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

Arithmetic has no implicit saturation or CPU fallback. Adding two unrestricted
byte images is rejected because the range reaches 510. Analysis is conservative:
even `combine(x,x,1,-1)` may fail without a proof accounting for correlations.
P coefficients fit [−128,127]; B combination coefficients fit [−64,63], since
they are doubled on the wire.

## Graph and optimization

`graph.py` defines immutable nodes, range propagation, helpers, and iterative
traversal. `lower.py` visits reachable expressions, eliminates structurally
identical expressions, removes simple identities, and folds a single shifted
read into its consumer's motion vector. It does not reassociate rounded or
clipped arithmetic.

Nested shifts cannot generally be added: `shift(shift(x,+1),-1)` loses its first
column at the intermediate boundary. The inner shift is materialized. Likewise,
`weighted(shift(x,1),bias=5)` gives 5 at the right edge, while
`shift(weighted(x,bias=5),1)` gives 0. Tests preserve both distinctions.

Disabling optimization retains explicit shifts and repeated work. Some such
graphs exceed the reference budget even when their optimized forms fit; they
fail with a resource error instead of silently changing semantics.

## Exact lowering

With denominator zero, weighted P prediction computes `clip8(w*x+offset)`.
Explicit weighted B prediction computes:

```text
clip8(((w0*a+w1*b+1)>>1) + ((o0+o1+1)>>1))
```

Weights `(2*wa,2*wb)` and equal offsets `(bias,bias)` therefore implement exact
`wa*a+wb*b+bias`. Graph range proofs make clipping inactive. Large affine biases
are split into legal offsets only if every intermediate also fits.

Threshold selection first searches legal P weights, denominators, and offsets.
A candidate must give 0 below the cutoff, 1 at it, and at most 1 at the largest
possible input. Monotonicity proves the intervening values. Negative results on
the false side may clip safely to zero.

For binary OR, the sum ranges from 0 to 2. The compiler finds
`(64*sum+64)>>7`, mapping 0,1,2 to 0,1,1 in one picture.

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

`schedule.py` computes last uses and drops dead values. Before an upcoming
instruction would lose a live picture at age 16, an exact P copy refreshes it.
References are resolved afterward; every emitted source is 1–16 pictures old.
At most 15 simultaneously live values are supported before an instruction.
More demanding graphs fail explicitly; this version has no spill or
recomputation strategy. Iterative graph traversal supports thousands of steps.

`backend.py` specializes dimensions, calls the existing shared bitstream writer,
caches the prediction tail, and writes the stream and manifest. `save(path,image)`
requires the initial pixel state, and rejects a path colliding with its sibling
JSON manifest. Dimensions must be multiples of 16 and satisfy the writer's
High-profile level-3.1 limits. Motion offsets are limited here to ±127 pixels.
Every picture retains a 16-pixel zero guard and neutral chroma; deblocking is off.

The manifest records logical graph IDs/ranges, physical instruction frame numbers,
reference dependencies, weights/offsets/motion, output frames, refreshes, and
optimization statistics. Folded nodes and aliases may have no standalone frame.
The prediction hash identifies reusable instructions; the stream hash also
includes the particular seed.

`verify.py` interprets graph semantics independently of lowering. `verify()`
compares every named output pixel with actual decoding. Extraction checks picture
count, zero guards, neutral chroma, and proven output ranges. No reference
evaluation occurs in `compile()`, `encode()`, `save()`, or `decode()`.

Binary computational outputs contain 0/1; previews scale them afterward. Maze
paths are extracted from decoded arrival times. MNIST reads one decoded class
bit per tile; input binarization and class-name formatting are host preprocessing
and presentation. Compilation uses only the Python standard library. Optional
checkpoint comparisons require the original PyTorch environment.
