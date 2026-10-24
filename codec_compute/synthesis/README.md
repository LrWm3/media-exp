# Discovering decoder computations

The shared compiler can now **search for an implementation**, verify every
scalar input, and lower the resulting plan through the existing codec backends.
There are no stored implementations of min, max, AND, OR, or XOR in the search
engine. Their mathematical specifications are ordinary opaque Python callables.

```python
import codec_compute as media

program = media.compile(
    media.minimum,
    target="hevc", width=16, height=16,
    input_ranges={"a": (0, 255), "b": (0, 255)},
)
evidence = program.synthesis_records[0]["search"]
assert evidence["verification"]["combinations_checked"] == 65536
inputs = {"a": [[13] * 16 for _ in range(16)],
          "b": [[7] * 16 for _ in range(16)]}
assert program.verify(inputs)["output"][0][0] == 7
```

`media.minimum`, `media.maximum`, `media.bit_and`, `media.bit_or`, and
`media.bit_xor` trigger search by default. The bit operations require **Boolean
0/1 images**; they are not bytewise bit operations. Use `synthesis=False` to
retain existing direct instruction selection for previously supported gates.
New min/max/XOR operations then have no direct implementation.

## What the prototype discovers

Default expression-tree bound: three primitive applications. Default candidate
budget: 50,000, including discarded signatures and work repeated after a
counterexample. The following instruction counts are observed discoveries,
not global optimality claims.

| Requested function | Domain | H.264 | HEVC | AV1 |
| --- | --- | ---: | ---: | --- |
| Minimum | Two bytes | 2 | 2 | No plan within bound |
| Maximum | Two bytes | 2 | 2 | No plan within bound |
| AND | Two Booleans | 2 | 2 | No plan within bound |
| OR | Two Booleans | 1 | 1 | 1 |
| XOR | Two Booleans | 3 | 3 | No plan within bound |

Each byte result is verified over all **65,536 pairs**; each Boolean result over
all **four pairs**. H.264 and HEVC software decoder tests also cover every byte
pair for the synthesized extrema and every Boolean pair for the gates. AV1 OR
runs through both libdav1d and libaom-av1. Hardware testing is deferred.

For example, the engine discovers minimum using two saturated subtractions,
and XOR using two opposite saturated subtractions and their saturated sum.
These examples describe search output, not templates supplied to the search.
The older hand-selected AND instruction is cheaper than the discovered plan:
this deliberately small primitive library does not yet expose every native
weighted-prediction coefficient or offset.

## Exact descriptions, not function-name recipes

`model.py` describes a primitive using signed integer weights, a divisor shift,
an offset, and final clipping:

```text
clip8(floor((sum(weights[i] * input[i]) + rounding_bias) / 2**shift) + offset)
rounding_bias = 0 when shift == 0, otherwise 2**(shift - 1)
```

Intermediate integers have unlimited precision. Division floors even for
negative values. The offset is applied after division; clipping happens last.
Descriptions are normalized into immutable data, so an overridden Python
evaluator cannot silently change the meaning bound by a verification record.

The initial H.264/HEVC libraries expose saturated addition, saturated
subtraction, and rounded average. AV1 exposes only rounded average. Input
aliases are free leaves. No constants or conditional branches are implicit.
These are **implemented pointwise subsets**, not complete codec capability
descriptions. Motion, borders, block geometry, input loading, reference buffers,
and bitstream serialization remain separate backend concerns.

The standalone engine accepts a new mathematical function without changing any
goal dispatch or instruction-selection code:

```python
from codec_compute.synthesis import library_for, synthesize, verify_record

spec = lambda a, b: abs(a - b)
result = synthesize(spec, library=library_for("hevc"))
assert result.found
assert len(result.plan.instructions) == 3
verify_record(result.plan, spec, ((0, 255), (0, 255)),
              library=library_for("hevc"), record=result.verification)
```

`Primitive` and `PrimitiveLibrary` can define other toy arithmetic libraries
for scalar experiments. `synthesis.lower.realize_plan` mechanically translates
a plan's supported descriptors into image IR. Supplying a custom descriptor
does not automatically provide a codec implementation: target lowering must
still support it exactly.

## Search and verification

1. Enumerate expressions by tree size using only input leaves and declared
   unary/binary primitives. Names are labels; the engine never inspects a
   specification's source, bytecode, or name.
2. Evaluate boundary and deterministic random samples. Retain the first
   smallest representative of each sampled output signature.
3. Independently enumerate the complete declared domain for every potential
   match. The verifier has its own indexed plan traversal and arithmetic
   evaluation; it does not call the search evaluator.
4. If verification finds a counterexample, add it to the samples and restart
   enumeration. If it passes, return the plan with its verification record.

This is bounded counterexample-guided enumeration. Specifications must be pure,
deterministic functions returning u8 integers (Boolean results normalize to
0/1). There is no Z3 dependency or symbolic proof claim.

Bounds count **expression-tree applications before structural sharing**. The
returned straight-line plan shares repeated subexpressions, so it can contain
fewer instructions. The prototype supports bounds from zero through eight.
`max_candidates` bounds evaluated candidates across all restarts; it is not a
wall-clock timeout. A specification's own execution time is outside its control.

`SearchResult.status` distinguishes `found`, `instruction_bound`, and
`candidate_budget`. Failure says nothing about a larger grammar, a larger bound,
or a codec's theoretical capabilities. No failed-result cache converts a bounded
miss into impossibility. Costs in primitive descriptions are metadata; the
current search orders by tree size, not weighted cost or decoder runtime.

Compiler options are `synthesis_max_instructions` and
`synthesis_max_candidates`. A `SynthesisError` carries the complete `.result`.
Scalar synthesis may succeed and target scheduling may still reject the graph,
for example when HEVC needs more than three simultaneously retained references.

## What a verification record establishes

Records include the exact ordered input domains, number of combinations,
primitive descriptions and fingerprint, indexed plan and fingerprint, complete
expected/actual output hashes, enumeration order, and any counterexample.
`verify_record` replays the domain and compares the entire record. A stored hash
or a mutable `verified` flag is not accepted as a substitute for replay.

Program manifests preserve both semantic and lowered graphs, map synthesis
requests to their source/operand/output node IDs, retain search evidence, and
include native scheduling costs and computation-tail hashes. Compilation
replays successful scalar records before replacing semantic operations.

Scalar records explicitly say `bitstream_verified: false`. They establish the
requested function under the primitive model on the recorded domain. The demo
records actual decoder validation separately, including decoder names and
output hashes. These finite software tests do not claim universal bitstream or
hardware correctness.

Composition keeps the original semantic graph for the reference interpreter.
Expanded primitives receive honest conservative intervals: XOR's independent
subtractions lose their correlation, so its lowered interval can be `[0,2]`
even though its exhaustively verified semantic range is `[0,1]`. Subsequent
search uses the original validated operand domains. A strict affine ancestor
stays strict whenever its recomputed interval permits; otherwise a saturated
realization is exact on reachable values because prior equivalence preserves
the original valid strict range. The compiler never writes fabricated narrow
intervals onto primitive nodes.

Encoding receives actual inputs only after compilation and appends a fixed
computation tail. It invokes neither synthesis nor the interpreter and never
uploads host-computed results.

## Reproduce

Use the dependencies documented in the [parent README](../README.md). All
commands run from the repository root:

```sh
python3 -m unittest discover -s codec_compute/tests -p 'test_synthesis*.py' -v
python3 -m codec_compute.synthesis.demo
```

The demo writes eleven successful streams and manifests plus a report covering
all fifteen target/goal combinations under ignored
`codec_compute/artifacts/synthesis/`. Its per-stream decoder check uses one
16x16 input pair; the test suite supplies the exhaustive decoded extrema checks.
Use `--target h264` or `--target hevc` without the optional AV1 serializer build.

Useful next steps are richer target primitive catalogs, measured target costs,
larger DAG search, and symbolic verification for wider/multi-input domains.
