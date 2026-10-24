# Exact two-input Boolean prediction minima

All minima below are **proved**, for the explicit model below. The exhaustive
single-instruction enumeration is an independently replayable certificate.
The multi-instruction lower bounds are structural proofs over unrestricted u8
intermediates; they are not conclusions from a bounded synthesis search.

## Model

The original inputs A and B are already resident and have literal values 0 or 1.
They vary independently at every pixel. A scalar operation applies uniform
parameters to one or two corresponding source values; it cannot inspect another
pixel, choose parameters from data, or receive a precomputed intermediate input.
Sources may be original inputs or earlier results, including repeated sources.
All intermediate values may use the complete integer range [0,255].

Each P instruction is

```
clip_u8(round_shift(w*x,d) + offset)
```

and each explicit weighted B instruction is

```
clip_u8(round_shift(w0*x+w1*y,d+1) + (offset0+offset1+1)//2).
```

Every weight and signalled offset lies in [-128,127]; d lies in [0,7]. Here
`round_shift(n,0)=n`; for positive s it equals
`(n+2**(s-1))//2**s`, using mathematical floor division, also for negative n.
Clipping occurs after rounding and offset addition. Residuals, deblocking,
fractional motion/interpolation, and precomputed intermediate inputs are absent.
These are legal explicit weighted-prediction operations, not the larger space
of every operation an H.264 decoder can perform.

Only original input aliases cost zero. Generating a constant costs one
instruction. There is one output value per input pair. The scalar cost is the
number of executed P/B operations in a straight-line computation. No free
selection, output negation, or host arithmetic is allowed.

The packed cost is the number of *computation pictures after input residency*.
Independent operations can occupy separate tiles of the same picture; each
operation reads only earlier pictures. Source planes can occupy separated tiles
of an input picture, with legal per-tile parameters supplied by slices.
Input upload, packing, and eventual readback are outside this count. These costs
are model minima, not claims that every backend schedule realizes them without
extra copy or reference-management pictures.

## Truth-table convention

Bits 0 through 3 describe `(A,B)=(0,0),(0,1),(1,0),(1,1)` respectively:
`mask=sum(f(A,B) << (2*A+B))`. Consequently A has mask 12 and B has mask 10.

| Mask | Function | Scalar minimum | Packed picture minimum |
|---:|---|---:|---:|
| 0 | FALSE | 1 | 1 |
| 1 | NOR | 1 | 1 |
| 2 | NOT A AND B | 1 | 1 |
| 3 | NOT A | 1 | 1 |
| 4 | A AND NOT B | 1 | 1 |
| 5 | NOT B | 1 | 1 |
| 6 | XOR | 3 | 2 |
| 7 | NAND | 1 | 1 |
| 8 | AND | 1 | 1 |
| 9 | XNOR | 3 | 2 |
| 10 | B | 0 | 0 |
| 11 | A implies B | 1 | 1 |
| 12 | A | 0 | 0 |
| 13 | B implies A | 1 | 1 |
| 14 | OR | 1 | 1 |
| 15 | TRUE | 1 | 1 |

## Zero instructions

Without instructions, the output must alias one of the two original inputs.
Thus exactly masks 10 and 12 have zero cost. Neither constant can be generated
for free. A P instruction with weight zero and offset 0 or 1 generates it.
Any successful one-instruction witness for another mask proves its scalar and
packed minima are both one.

## Exhaustive single-instruction certificate

### Effective B offsets

For signed offsets o0,o1, `floor((o0+o1+1)/2)` always lies in [-128,127]. Every
integer c in that interval is obtained with o0=o1=c. Thus using one effective
signed offset loses no B operation, even when the two signalled offsets differ.
The verifier also enumerates all 65,536 raw offset pairs and checks this range.

### Exact offset elimination

Fix sources, weights, and denominator. Let r be the four rounded values before
offset addition, and M=max(r). If the final output is binary and contains a 1,
some unclipped value must be exactly 1: upper saturation at 255 cannot produce
1. Since clipping is monotone, `clip_u8(M+c)=1`, hence necessarily `c=1-M`.
With this c, exactly the argmax positions become 1; every other integer rounded
value becomes at most zero and clips to zero. This argument also covers the
constant-one output. It exhausts every binary output containing a 1 for that
fixed parameter tuple, rather than sampling offsets.

The all-zero input pair has rounded value zero, so M>=0. Binary inputs and
legal weights give M<=127 for P and B, even for repeated sources; hence
`1-M` lies in [-126,1] and is always a legal signed offset. Constant zero is
covered separately by its explicit weight-zero P witness. No other binary
outputs exist for any skipped offset.

### Exact source reductions and counts

* P has two choices of original source, 256 weights, and eight denominators:
  **4,096 classes**.
* B with distinct sources is represented by ordered sources `(A,B)`, all
  256 squared weight pairs, and eight denominators: **524,288 classes**.
  The source order `(B,A)` produces exactly the same operations by exchanging
  the two weights; effective offset addition is symmetric too.
* B with repeated sources depends on the sum of its two weights. Every sum
  from -256 through 254 is legal: use
  `w0=max(-128,total-127)` and `w1=total-w0`. With two source choices and eight
  denominators, this gives **8,176 classes**. Different pairs with the same sum
  are arithmetically identical before rounding and clipping.

Total: **536,560 exact classes**. Before these reductions, the grammar contains
1,048,576 P parameterizations and 536,870,912 B parameterizations when effective
offsets are counted once. Raw B offset pairs add redundancy, already removed by
the proved effective-offset equivalence.

`atlas.py` enumerates every reduced class, calculates all four rounded values,
records the argmax-mask histogram, and retains a legal witness for each mask.
The complete one-instruction mask set is
`{0,1,2,3,4,5,7,8,10,11,12,13,14,15}`. Every retained witness is replayed on all
four input combinations by a separate instruction oracle. The certificate
includes exact class counts, histograms, source/offset reductions, effective
offset coverage, witnesses, and a SHA-256 digest of its canonical JSON payload.
The digest checks reproducibility; the enumeration and proofs establish
correctness.

## No one-instruction parity

A scalar weighted prediction is monotone in each individual operand: increasing
an operand can only increase its weighted sum for a positive coefficient,
decrease it for a negative coefficient, or leave it fixed for zero. Rounding,
offset addition, and clipping preserve this order. Repeated sources combine
their weights and retain the same property.

A Boolean function with a fixed monotonic direction in each original input is
called unate. XOR and XNOR are not monotone in either input in any fixed
direction: changing B flips the result one way at A=0 and the opposite way at
A=1, and symmetrically for A. Therefore neither can be a one-instruction
output. This proof also checks the exclusion independently of enumeration.

## No two-instruction parity

Let the first result be h(A,B), with arbitrary u8 values. It is monotone in each
input, with a fixed direction, by the preceding lemma. If the second instruction
does not use h, its output is already a single-instruction function. If the
output aliases an earlier value, that earlier value suffices and cannot be
parity either. Otherwise there are only these source cases:

1. **h alone or h twice.** The second result is a monotone unary function of h;
   repeated h sources simply sum their weights. Its composition with h is
   therefore monotone in both A and B, with fixed directions.
2. **h and A.** Fix A. As B varies, the A operand is fixed and the second
   instruction is monotone in h, with a direction determined only by h's
   coefficient. Since h's monotonic direction in B is also fixed independently
   of A, the output's direction in B cannot reverse with A. It may become flat
   because of clipping or rounding, but cannot have opposite nonzero directions.
3. **h and B.** By the symmetric argument, the output has a fixed monotonic
   direction in A independently of B.

XOR and XNOR require reversals in both inputs and contradict every case.
This covers the full instruction grammar, including negative coefficients,
all legal denominators, repeated sources, a constant or nonbinary h, and both
clipping boundaries. There is no restriction to binary intermediates and no
finite multistep search hidden in this argument. Thus each parity mask requires
at least three scalar instructions.

Three suffice. Define `B(x,y;u,v,c)` as a B instruction at denominator zero,
so its result is `clip_u8((u*x+v*y+1)//2+c)`:

```
XOR:
  h0 = B(A,B;  2,-2, 0) = max(A-B,0)
  h1 = B(A,B; -2, 2, 0) = max(B-A,0)
  out = B(h0,h1; 2,2,0) = h0+h1

XNOR:
  h0 = B(A,B;  2, 2,-1) = max(A+B-1,0)
  h1 = B(A,B; -2,-2, 1) = max(1-A-B,0)
  out = B(h0,h1; 2,2,0) = h0+h1
```

In each case h0 and h1 are binary, mutually exclusive, and independent. Their
sum is exactly the requested binary output. Every listed parameter is legal.

## Packed pictures

With zero computation pictures, only original input aliases are available.
With one computation picture, each output tile can only read original input
tiles, so each output is a single instruction regardless of how many independent
neighboring tiles the picture computes. It cannot read another tile computed
in that same picture. Thus parity cannot fit in one computation picture.

For either parity witness, picture 1 computes h0 and h1 in separate tiles;
picture 2 reads those tiles and computes their sum. Source duplication in the B
reference lists permits selecting the two tiles from the same reference
picture. Therefore parity needs exactly two packed computation pictures.
Every other non-alias function has a single-instruction witness and needs
exactly one picture. `packed_schedule` records the result indices in each
picture, and tests verify that all reads reference strictly earlier pictures.

## Replaying the certificate

From this directory, using Python 3 with no additional dependencies:

```sh
python atlas.py --output /tmp/boolean-atlas.json
python atlas.py --verify /tmp/boolean-atlas.json
python -m unittest test_atlas.py
```

Verification regenerates the exhaustive certificate and replays all 64 Boolean
truth-table cases, checks that the non-unate masks are precisely XOR and XNOR,
and compares every saved plan and claimed minimum with the proof-backed atlas.
Altering a certificate count, witness, offset range, or minimum is rejected.
The verifier does not formalize the structural proofs in a theorem prover;
those proofs are fully stated above and apply to every permitted u8 program.
