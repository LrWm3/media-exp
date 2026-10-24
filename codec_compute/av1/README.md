# AV1 decoder-compute subset

This backend emits real AV1 in IVF. A pinned libaom serializer generates a
**reusable instruction tail at compile time using zero input placeholders**.
At execution, it serializes only supplied input planes losslessly, then appends
that exact tail. FFmpeg/libdav1d or FFmpeg/libaom-av1 performs the computation.
No host arithmetic output is computed, uploaded as a replacement input, or
encoded into residual coefficients.

```sh
python3 -m codec_compute.av1.bootstrap
python3 -m unittest codec_compute.tests.test_av1 -v
```

The explicit bootstrap needs Git, CMake, a C/C++ compiler, and network access.
It clones libaom **v3.12.1**, verifies commit
`10aece4157eb79315da205f39e19bf6ab3ee30d0`, applies [libaom.patch](libaom.patch),
and builds the small [bridge.c](bridge.c) driver. Source, library, executable,
and logs stay in a content-addressed `$XDG_CACHE_HOME/codec-compute/av1` directory
(default `~/.cache`). No upstream source tree or binary is committed. Python
uses only its standard library; decoding additionally requires FFmpeg with one
of the two supported AV1 decoders. Compilation never downloads or builds tools
implicitly.

```python
from codec_compute.av1 import AV1Step, compile_program

program = compile_program(16, 16, ("left", "right"), (
    AV1Step("mean", "average", ("left", "right")),
    AV1Step("shifted", "translate", ("mean",), dx=1, dy=-1),
    AV1Step("output", "average", ("shifted", "left")),
))
inputs = {"left": [[10] * 16 for _ in range(16)],
          "right": [[21] * 16 for _ in range(16)]}
stream = program.encode(inputs)             # Input serialization + fixed tail.
execution = program.decode(stream)          # Real libdav1d decode.
assert execution.frames["output"][0][0] == 13
assert stream.endswith(program.compute_tail)
```

`AV1Program.frame_names` names every displayed frame, including inputs and
intermediates. `encode_inputs(inputs)` returns only an IVF input prefix;
`encode(inputs)` returns a complete IVF stream without decoding;
`decode(stream, decoder="libdav1d")` returns an `AV1Execution`; and
`run(inputs, decoder=...)` combines encode and decode. Input values may be
row lists/tuples of integers or row-major bytes. `AV1Execution.frames` contains
ordinary Python lists extracted from decoder output. Native serialization has a
180-second timeout; decoding has a 60-second timeout. FFmpeg runs with `-xerror`,
and any error-level stderr or nonzero process status fails explicitly.

## Exact supported semantics

| Native operation | Mathematical result | AV1 implementation |
| --- | --- | --- |
| `copy` | `out[y,x] = source[y,x]` | Single-reference integer prediction, zero MV |
| `translate` | `source[clamp(y+dy), clamp(x+dx)]` | Single-reference integer motion vector |
| `average` | `(left[y,x] + right[y,x] + 1) // 2` | `COMPOUND_AVERAGE` with two zero-MV references |

These operate on unsigned 8-bit monochrome planes. Average rounds upward at
half-integers; it does **not** represent exact addition, floor average, or an
arbitrary signed affine expression. Translation uses edge replication. Width
and height must be multiples of 16 between 16 and 4096, inclusive. Translation
components are integers between -16 and 16. Other operations, dimensions,
displacements, and non-u8 inputs fail explicitly.

The compiler assigns reference lifetimes to AV1's eight reference slots, and
reuses a slot once its old value has no remaining consumer. A destination may
replace a source after the current frame reads it. There may be at most eight
initial input planes. Programs whose live sources leave no slot for their next
result fail instead of spilling values to host-computed images. There is no
stage-count limit from the reference allocation: the decoder tests include
20 composed averages and expose all 22 displayed input/intermediate frames.
Two identical average sources simplify to copy.

Unsupported here: arbitrary weighted or signed affine prediction, saturating
addition/subtraction, threshold and bitwise operations, arbitrary tile gathers,
fractional motion, alternate borders, resizing, chroma computations, and
higher bit depths. AV1 has additional tools; this module claims only the
implemented and tested subset.

## Why decoder computation is genuine

The patch has three narrow changes to the reference encoder:

1. Fix partitioning to 16x16 blocks and enable compound syntax only when needed.
2. Serialize input frames with ordinary lossless intra blocks. The first input
   is a keyframe; subsequent inputs refresh selected slots without resetting
   every reference.
3. For instruction frames, bypass mode search after reference-context setup:
   select `NEWMV` for copy/translation or `GLOBAL_GLOBALMV` for averaging, set
   compiler-chosen MVs/references, select `COMPOUND_AVERAGE`, and force
   `skip_txfm=1`. No operation residual is transmitted.

[bridge.c](bridge.c) disables CDEF, restoration, loop filtering, global-motion
transforms, reference-MV prediction, order hints, and CDF adaptation. Each frame
starts with no inherited primary-reference entropy context. The bridge uses
libaom's external reference configuration to select source slots and refresh
exactly the destination slot. During compilation its internal reconstructions
contain only neutral placeholders. During execution Python calls it for input
frames **only**. The compute packet bytes are immutable across real inputs.

Tests independently compare all decoded samples against arithmetic written only
in the tests. They cover 0/255, ramps, alternating extrema, random inputs, odd
average sums, positive/negative edge translations, multiple blocks, repeated
reference-slot reuse, and changed inputs. The 256x256 exhaustive case covers all
65,536 possible input pairs for rounded average through both libdav1d and
libaom-av1. An eight-input retention case keeps all eight reference slots live,
replaces each only after its final read, and validates every displayed result
through both decoders. It also verifies that a complete encode with real input
bytes exactly matches the separately generated input prefix plus neutral-compiled
tail. FFmpeg `trace_headers` confirms zero quantizers, disabled CDF adaptation,
no inherited primary entropy context, disabled filtering/order hints/frame IDs,
and each precise one-bit reference refresh mask. A serializer spy checks that
production runtime encoding receives input instructions only.

The normative basis is the AV1 specification's inter-prediction and compound
prediction processes, including `COMPOUND_AVERAGE` rounding:
[AV1 Bitstream & Decoding Process Specification](https://aomediacodec.github.io/av1-spec/).
The patched encoder is pinned to
[libaom v3.12.1](https://aomedia.googlesource.com/aom/+/10aece4157eb79315da205f39e19bf6ab3ee30d0/).
