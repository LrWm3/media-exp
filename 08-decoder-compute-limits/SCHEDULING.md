# Instruction synthesis and picture scheduling interact

Two XOR programs both attain the proved minimum of three scalar instructions:

```text
Parallel construction:                 Serial construction:
u = clip(A-B)                           g = A+B
v = clip(B-A)                           h = clip(g-1)
out = u+v                               out = g-2*h
```

All operations use exact legal P/B parameters. In the serial construction,
`g` can be 2: the instruction model permits nonbinary intermediates. XNOR has
analogous parallel witnesses from the atlas and the serial output `1-g+2*h`.

The parallel construction has two dependency layers and can share its first
picture across two tiles. The serial construction has three layers; packing
cannot remove those dependencies. Choosing only by scalar instruction count
therefore loses information needed by the scheduler.

## Emitted costs at 32×32

All rows execute three scalar instructions. Picture and luma counts below
cover predictions only; two additional PCM pictures load the inputs.

| Construction/schedule | Computation pictures | Coded luma samples | Slices | XOR tail bytes | XNOR tail bytes | Reference slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parallel, packed | **2** | 14,336 | **20** | **466** | **468** | **2** |
| Parallel, unpacked | 3 | **12,288** | 24 | 515 | 518 | 3 |
| Serial, packed | 3 | **12,288** | 24 | 484 | 485 | **2** |

The serial graph has one ready operation per layer, so its packed layout uses
one tile. Its coded dimensions are 64×64, compared with 112×64 for the parallel
packed graph. Including inputs, the streams process 20,480 and 28,672 luma
samples respectively. Final I420 readback costs 6,144 versus 10,752 bytes.

Among these candidates, serial packed dominates parallel unpacked on the five
recorded prediction-cost metrics. Parallel packed and serial packed both remain
on the Pareto frontier: fewer pictures/slices/bytes versus fewer coded samples.
The reference allocator also needs two slots for serial packed instead of three
for parallel unpacked. Instruction count alone predicts none of these details.

This is a bounded comparison of **two supplied constructions and three schedules
per function**, not an exhaustive joint optimizer. The proved function-wide
minimum remains two packed computation pictures; no global optimality is
claimed for bytes, coded area, slices, latency, or reference slots.

## Validation and reproduction

[joint.py](joint.py) emits all six candidates. FFmpeg, OpenH264, and VA-API
match every decoded byte and the independent truth oracle on both a truth
pattern and seeded random inputs. NVDEC validates the truth-pattern stream for
each candidate: **28 pictures, 208,896 I420 bytes**. No timing comparison is
claimed for this small single-evaluation set.

[Candidate plans and costs](assets/joint.json) ·
[NVDEC validation](assets/joint-nvdec-results.json).
The candidate report records source hashes and its commit. Its dirty-tree flag
reflects documentation/capture additions during preparation; all hashed source
files match their committed versions.

```sh
python3 08-decoder-compute-limits/joint.py
python3 08-decoder-compute-limits/benchmark.py \
  --bundle 08-decoder-compute-limits/out/joint --backend nvdec \
  --helper decoder_core/.build/nvdec_decode \
  --output 08-decoder-compute-limits/out/joint-nvdec
```

This gives the next optimizer a concrete objective: retain alternative exact
instruction graphs, schedule each with real tile/reference costs, and choose
using measured runtime costs. The current experiment establishes why that
choice matters; it leaves the general search and compiler integration open.
