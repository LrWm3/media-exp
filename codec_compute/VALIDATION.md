# Validation record — 2026-10-24

Environment: Python 3.12.13; FFmpeg 6.1.1-3ubuntu5; pinned libaom serializer
v3.12.1 (`10aece4157eb79315da205f39e19bf6ab3ee30d0`). The AV1 serializer
was built using the documented bootstrap and its second invocation reused the
cached executable. Video decoders are unmodified.

## Prediction bands and decoder interfaces

[Experiment 09](../09-codec-compute-ir/README.md) is now the numbered shared-IR
entry point. Experiment 05's frontend is deprecated for new kernels; its
backend and legacy packing remain required and available.

[Experiment 10](../10-decoder-interface-limits/README.md) adds opt-in H.264
prediction row bands with the actual cross-row motion-vector predictor.
One row remains the default with byte compatibility. Compatible 512×512
predictions use 34/9/1 slices for one-row/four-row/whole-picture layouts;
heterogeneous headers and nonzero offsets retain the earlier layout. Tests
cover exact P/B fractional samples, independent fields, all denominators,
retained references, partial bands, legal vector endpoints and default bytes.
Completed full suites pass **187 shared tests and 105 decoder-core tests**;
the subsequently added endpoint regression passes in the **nine-test band
suite**. Protocol-v2 diagnostic coverage passes **17 session tests**, and the
paired experiment tooling passes **15 focused tests**. These counts overlap;
they are not a combined test total.

The frozen 512×512 export contains **72 streams / 1,428 pictures**, with
**216 local decoder checks** and **48 byte-identical complete-I420 pairs**.
NVDEC preflight checks **1,356 real pictures / 601,933,824 I420 bytes** plus
72 fences. All **300 timing trials** pass. Completion-only timing returns
zero copied pictures; its 360 new postflight requests check another **6,780
real pictures**, outside timing. They do not inspect pixels from the earlier
completion-only requests. Final close validates all decoded/copied counts and
drains every pending fence. Source, bundle, CPU/PCM build and worker hashes
bind the results; a curated six-case fixture-0 capture preserves actual pixels.

For 64 stages, whole-picture slices reduce warm native-input/host-output
elapsed time **216.27 → 96.82 ms** versus one-row slices. CPU takes **81.86 ms**;
its five-trial range remains entirely below the decoder's range. Decoder
process CPU is **47.48 versus 81.71 ms**, a **41.9% reduction** at an **18.3%
latency penalty**. At 32 stages process CPU is 33.8% lower. These synthetic
chains establish a resource tradeoff, not an application or energy win.

Atlas and interpolation remain slower than CPU even with prepared input and
completion-only output. All modes still decode input pictures and submit an
IDR fence. Their separate noisy medians are not isolated stage-cost estimates;
some omitted-work medians are higher. No reference-surface import, GPU consumer,
retained state across jobs, direct-array API execution or hardware modification
was tested. The read-only availability probe found no loadable NVDECODE library
on this software path; this does not establish hardware impossibility.

## Zero-guard merging and CPU offload

The shared compiler now enables compatible zero-guard row merging by default;
the direct Writer preserves its previous default. Eligible 512×512 predictions
fall from **98 to 34 slices**, while nonzero offsets, heterogeneous headers and
macroblock mode retain the old layout. Regression suites pass **182 shared
compiler tests and 89 decoder-core tests**, plus three paired-runtime and
thirteen concurrency tests. Existing level/resource checks remain conservative.

The [paired experiment](workloads/guard_fusion/README.md) freezes six recurrence
depths (1, 4, 8, 16, 32 and 64), eight changing inputs and both slice layouts.
Its **96 streams / 2,192 pictures** pass **288 local decoder checks**, with
48 byte-identical paired I420 sequences. NVDEC verifies all **2,096 real
pictures / 930,422,784 I420 bytes**, plus 96 counted completion fences. The
committed capture subset retains twelve streams, 274 pictures and six pairs.

The isolated report preserves **90 timing trials** against a CPU binary
identical to the previous native/unrolled tuning winner. For 64 stages,
merging lowers median decoder elapsed time **518.23 → 205.67 ms** (60.3%) and
process CPU **71.13 → 49.85 ms** (29.9%). Relative to native CPU, it uses
**39.1% less process CPU** but takes **2.49× longer**. Depth 32 is the earliest
tested point with separated CPU-time ranges: **29.19 versus 41.00 ms**, a
28.8% reduction. Depth 16 is inconclusive; every tested depth still loses CPU
elapsed time. Setup, warmup and close are recorded outside warm timings, and
process CPU excludes IRQs and kernel workqueues. Energy is unmeasured.

A further **20 fixed-window contention trials** share one logical CPU among
foreground work, helper threads and checked PBKDF2 background work. At 1 Hz and
a 750 ms deadline, CPU and merged NVDEC each complete **60/60 requests within
deadline**; baseline NVDEC misses **4/60**. Merged background throughput is
**920.50 versus 903.75 chunks/s** by median, a **1.85% observed gain**; all five
repeat-paired gains are positive, but absolute ranges overlap and the control
varies. Treat this as a small synthetic benefit requiring longer application
validation. Merged foreground process CPU is 37.6% lower under contention,
while median response is slower (287.18 versus 158.64 ms). No jobs are dropped
or complete beyond the observation window. Final fence counters, source hashes,
thread affinity/scheduling and restored runner affinity all pass. Protocol
success remains explicitly separate from deadline qualification.

## Matched native CPU and NVDEC comparison

The [CPU comparison](workloads/cpu_comparison/README.md) adds independent native
CPU baselines for copy, averaging, patch atlases, fractional interpolation, and
16/64-stage retained-image recurrences. Optional native I_PCM serialization
matches the original Python writer byte-for-byte. All timed results are checked,
process CPU is measured separately from elapsed time, and the live helper's CPU
clock is read directly instead of relying on reaped-child accounting.

The initial 128×128/256×256 corpus passes **96 streams, 1,552 pictures and 288
local decoder checks**. Live NVDEC checks all **1,456 real pictures / 118,530,048
I420 bytes**, plus counts 96 discarded completion fences. The report preserves
**270 timing trials**, source/build/binary hashes and a full input index. Every
case loses to the native CPU in both measured metrics, including native uploads,
prepared inputs and finite 64-job batching. See its README for the exact scope.

The larger source-pinned follow-up passes **48 streams, 776 pictures and 144
local checks**. NVDEC verifies all **728 real pictures / 323,162,112 I420 bytes**,
plus counts 48 discarded fences. Its report contains **160 further timing
trials**. Together the two corpora cover 144 hardware jobs, 2,328 decoded
pictures, 2,184 complete real-picture readbacks and 430 timing trials. There is
no decoder elapsed-time win. The 512×512/64-stage synthetic recurrence consumes
14.6% less measured host process CPU than the shared-filter portable baseline,
but finishes 6.19× slower; other cases lose both metrics. A further **60 CPU-only
trials** test portable, native and native/unrolled builds of both CPU kernels.
The narrow gap survives: **70.078 ms NVDEC versus 81.703 ms CPU** process time
(14.2% less), but **509.032 versus 81.840 ms** elapsed (6.22× slower). All 636
initial/warmup/timed tuning calls match exact outputs. These findings do not
establish energy savings or application-throughput gains.

This milestone's shared compiler suite passes **178 tests**; decoder-core passes
**82 tests**, with no skips. Explicit High level 5.1 adds normative resource
checks and enables arbitrary 512×512 PCM at the tested 30 fps. Default level 3.1
bytes remain unchanged. Spatial field validation now uses the writer's true
motion limits rather than the frontend's smaller historical bound. Nine tests
cover a stronger CPU recurrence which shares separable half-sample calculations;
three integration tests compare all six workloads against the IR and FFmpeg.
The generic CPU kernel's seven tests and the native serializer's thirteen tests
also pass. Native address/undefined-behavior checks cover the new serializers
and recurrence kernel; the generic CPU tests pass under UBSan.

## Larger inputs and resident decoder requests

The complete shared compiler suite passes **172 tests in 42.737 s, with no
skips**. The decoder-core suite passes **68 tests in 5.672 s**. New checks cover
explicit High level 4.1 limits, arbitrary 256×256 inputs, constant intra IDR
fences, socket framing, worker failure/timeout handling, selected-frame guard
and chroma validation, named aliases, and runtime execution without host graph
evaluation or instruction emission. Default level-3.1 stream bytes remain
unchanged for the validated historical fixtures.

The [resident experiment](workloads/resident/README.md) exports **64 streams,
256 pictures, and 20,840,448 full I420 bytes**, with **192 local checks** through
FFmpeg, OpenH264 and VA-API. Inputs range over scenes, noise, zero and 255 at
128×128 and 256×256. Baseline/fused interpolation and matched unwarped averaging
use identical inputs; passthrough is a separate single-input control. Both the
semantic interpreter and independent task formulas agree with every named output.
The committed representative subset preserves eight streams and 32 pictures.

The live NVDEC path requires a subsequent IDR to release a short reference
chain. The runtime therefore appends one discarded, constant-128 IDR per job:
**149 bytes at 128×128 and 372 bytes at 256×256**, including the coded guard
geometry. Input IDR identifier 0 and fence identifier 1 alternate correctly.
The fence costs a real decoded picture and resets references between jobs.
Each request completes before the next submission; only shutdown sends EOS.
Successful close validates delayed diagnostics and drains the final fence.
The ordinary saved program and its compute-picture count remain separate from
this explicitly counted runtime overhead.

NVDEC validates all **64 changing jobs** through eight live sessions, including
both shapes and every baseline/control. It decodes **256 pictures including
64 fences**, and reads back every one of the **192 real computation/input
pictures** for exact I420 comparison. Discarded fences are counted and checked
for decoder errors but are not copied back by the resident path. Earlier native
probes also rejected malformed IDR boundaries and overflowing dimensions, and
confirmed compatibility with the existing finite-file CLI.

The [complete NVDEC report](workloads/resident/assets/nvdec.json) passes five
repeats of all five modes and eight cases. For fused interpolation, fresh-session
host-input medians are **297.40/342.22 ms per request** at 128×128/256×256;
warmed resident medians are **29.05/78.99 ms**. Warm pre-encoded medians are
**10.60/15.17 ms**; input serialization alone measures **19.40/65.15 ms**.
These values are median eight-request trial durations divided by eight. Fresh
requests include startup and shutdown; resident timings exclude creation,
one complete warmup and close, which are recorded separately. Including those
phases within each resident host trial gives **64.97/132.82 ms per timed request**
with setup/warmup/close amortized over eight requests. No decoder-only or CUDA
speedup is inferred. Each request uploads fresh inputs and decodes one fence.

One earlier run passed full correctness and its first timing repeat, then failed
at cold-session shutdown. Its original wrapper lost the native diagnostic; the
cause remains unknown. The runtime now preserves failure diagnostics and exit
status. The subsequent complete benchmark passes, but does not establish that
the original shutdown issue is fixed. The experiment archives the interrupted
report separately and excludes its partial timings from these conclusions.

```sh
python3 -m unittest discover -s codec_compute/tests -v
python3 -m unittest discover -s decoder_core -p 'test_*.py' -v
python3 -m codec_compute.workloads.resident.prepare --sizes 128 256 --jobs 8
```

## Block routing and codec-defined fractional sampling

At this milestone, the complete suite passed **165 tests in 34.244 s, with no skips**.
New semantic and target tests cover integer block gathers, all sixteen H.264
fractional luma phases, negative coordinates, diagonal filter precision, image
boundaries, immutable fields, and conservative range proofs. Target tests also
exercise two independently moving B-prediction sources, retained references,
named outputs, successive crop/filter barriers, multistage arithmetic, and
cost-based fallback. Changed inputs reuse identical prediction tails with host
evaluation and synthesis disabled during execution.

The [resampling experiments](workloads/resampling/README.md) cover patch routing,
motion-compensated interpolation, bandwise correction, compensated absolute
residuals, dense polarity maps, and three stereo candidate costs. At 128×128
logical pixels, their fused computation-picture counts are **1, 1, 1, 3, 6,
and 9**, respectively. Interpolation falls from three computation pictures to
one; both spatial reads and their rounded blend execute in that picture.

The complete export contains **113 streams and 588 pictures**, checked through
FFmpeg, OpenH264, and VA-API in **339 local decoder checks**. All 162 named outputs
match both the semantic interpreter and independent task formulas. Full I420
comparisons also check guard luma and chroma. The saved capture binds each output
panel to its decoded frame; analytic truth is labeled separately.

A controlled nonlinear texture translated by ±2.5 pixels gives a known-motion
quality witness: compensation lowers midpoint mean absolute error from
**14.56 to 0.06 grayscale levels** on an interior 112×112 crop. This establishes
the result for the supplied-motion fixture, not general optical-flow or
occlusion quality. Motion fields are fixed at compilation. Patch routing packs
a 64×64 active ROI within the existing surface; it does not shrink the 160×160
coded geometry or readback allocation. The current uploader rejects the tested
256×256 inputs at its level-3.1 I_PCM access-unit bound.

NVDEC passes the same **113 streams, 588 pictures, and 22,579,200 I420 bytes**,
including both decoded quality-witness outputs. Its actual output pixels
reproduce the reported midpoint errors. Both timing runners validate the full
corpus before measuring batches of one and eight jobs, with one warmup and five
repeats per case/mode. Raw reports preserve samples and source/artifact hashes.

For eight-job NVDEC batches, interpolation's directly measured host-input-to-output
median falls from **533.35 to 490.91 ms per batch** (66.67 to 61.36 ms per job),
about **8%** in this run. Pre-encoded selected-output medians fall from 368.41
to 342.99 ms per batch; the matched unwarped-average control measures 343.01 ms.
These finite batches include process/pipeline startup, I/O, readback, and exact
checks. Input serialization alone measures 164.43 ms per eight interpolation
jobs in its separate mode. Independent mode medians must not be subtracted to
infer decoder-only time. The local FFmpeg sample ranges overlap substantially;
they do not establish a reliable latency improvement. No GPU arithmetic baseline,
concurrent GPU load, or persistent-stream performance is measured. FFmpeg and
NVDEC timings come from different hosts and are not a controlled backend race.

```sh
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.workloads.resampling.demo --size 128 --jobs 8
```

## Spatial read fusion and temporal benchmarks

At this milestone, the complete suite passed **133 tests in 29.709 s, with no skips**.
Sixteen fusion tests cover zero borders, nonzero input minima, successive crops,
named aliases, shared reads, threshold stages, synthesized extrema, motion limits,
and fallback after either schedule or stream-cost validation fails. Four
benchmark protocol tests check independent references, selected readback,
copy accounting, and artifact validation.

The [spatial comparison](workloads/motion-fusion/README.md) validates both plans
and both input fixtures: **12 saved streams, 136 pictures, and 36 local checks**
through FFmpeg, OpenH264, and VA-API. Dilation and smoothing fall from eight to
four computation pictures; local min/max/range falls from 21 to 17. Named
outputs remain exact. These are static cost reductions; spatial latency was
not measured. NVDEC also passes all **12 spatial streams: 136 pictures and
835,584 full I420 bytes**, matching the software-decoded reference exactly.

The [temporal benchmark](workloads/temporal/README.md) measures four tasks at
128×128 logical pixels using batches of one and sixteen independent jobs.
Input serialization, pre-encoded full/selected readback, and a directly timed
host-input-to-output path have separate measurements. Every task uses the same
sixteen deterministic input pairs, and every returned picture is checked
exactly. Reports include raw samples, medians, costs, counters, source hashes,
and the precise timing boundary. Finite batch throughput includes startup;
it does not establish persistent-stream latency or concurrent GPU performance.

Each temporal backend, FFmpeg and NVDEC, first validates **64 streams, 320
pictures, and 12,288,000 I420 bytes**, plus all 112 named output planes.
Every measured trial also checks all returned bytes. On the NVDEC benchmark
host, sixteen-job batches measure **43.6–54.0 ms per job** for the complete
host-input-to-output path, including roughly **19.5–20.2 ms per job** for the
separately measured serialization mode. The two medians must not be subtracted
to infer decoder-only time. Selected readback reduces copied pictures without
a consistent timing improvement. FFmpeg timings come from a different host;
these measurements are not a controlled software/hardware speed comparison.

```sh
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.workloads.fusion
python3 -m codec_compute.workloads.benchmark
```

## Shared-IR workload survey

At this milestone, the complete suite passed **113 tests in 25.769 s, with no skips**.
The new `absdiff` semantic operation is synthesized for H.264 and HEVC, verified
over all 65,536 byte pairs, and decoded across the complete byte-pair domain on
both targets. It uses three computation pictures. The AV1 average-only library
reports a bounded search miss. Runtime purity, forged ranges, aliases, and
composed thresholds are tested independently.

Nine task graphs are attempted across three targets. **14 combinations run**;
the remaining 13 report specific backend restrictions or bounded search misses.
Every accepted program executes two input fixtures with an identical prediction
tail: **28 streams, 62 local decoder runs, and 45,056 named output pixels**.
Saved-stream outputs agree with both the IR interpreter and independent task
formulas. New tests also cover cross-shaped neighborhoods, directional changes,
zero/clamp borders, and observable nested rounding.

**NVDEC passes all 14 H.264 streams: 120 pictures and 737,280 I420 bytes.**
This includes both fixtures for all supported H.264 tasks. HEVC and AV1 hardware
execution remains untested. Decoder latency was not measured; suite duration
and serialized byte counts are not throughput benchmarks.

The [workload report and captures](workloads/README.md) retain source, serializer,
stream, and output hashes; static costs; input-prefix sizes; and rejection
reasons. Reproduce with:

```sh
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.workloads.demo --no-h264-fuse-shifts
```

The sections below preserve earlier milestone results.

## Automatic synthesis extension

The initial synthesis milestone passed **93 tests in 21.312 s, with no skips**. It added
43 synthesis tests: 13 search, 17 independent verification, and 13 compiler/
software-decoder integration tests. This run used software decoders only;
hardware validation was deferred at that milestone.

```sh
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.synthesis.demo
```

The generic engine discovers full-byte min/max and Boolean AND/OR/XOR for H.264
and HEVC, and Boolean OR for AV1. Every successful plan has an exhaustive scalar
record: 65,536 states for each byte extremum and four states for each Boolean
gate. The remaining four AV1 goals report `instruction_bound` for the initial
average-only library at three tree applications; this is not an impossibility
claim about AV1.

The integration suite decodes **all 65,536 byte pairs** for both synthesized
extrema on H.264 and HEVC, and all four Boolean pairs for the gates. AV1 OR
passes both libdav1d and libaom-av1. Other checks cover opaque callable goals
(including full-byte absolute difference), counterexample refinement, budget
exhaustion, descriptor-based arithmetic independent of evaluator overrides,
signed rounding, tampered-record rejection, composed Boolean ranges, existing
strict affine lowering, backend resource rejection after scalar success, and
changed inputs with no runtime synthesis or host computation.

The synthesis demo writes eleven streams and their manifests, plus a report
covering fifteen target/goal combinations, under
`codec_compute/artifacts/synthesis/`. Its per-stream software check uses one
16x16 pair; exhaustive decoded byte coverage belongs to the test suite.

| Discovered program | Target | Compute pictures | Prediction bytes |
| --- | --- | ---: | ---: |
| Min, max, or Boolean AND (each) | H.264 | 2 | 212 |
| Boolean OR | H.264 | 1 | 105 |
| Boolean XOR | H.264 | 3 | 318 |
| Min, max, or Boolean AND (each) | HEVC | 2 | 32 |
| Boolean OR | HEVC | 1 | 15 |
| Boolean XOR | HEVC | 3 | 47 |
| Boolean OR | AV1 | 1 | 27 |

Each program also loads two input pictures. Counts reflect this small grammar,
not optimal target costs. Scalar evidence and software-decoder evidence remain
separate in the report. See [synthesis details](synthesis/README.md).

## Original multi-codec milestone results

| Check | Result |
| --- | --- |
| Existing H.264 compiler/native/interpolation/scheduler baseline | 33 tests passed, 5.877 s |
| Combined new native-backend, IR, and cross-target suite | 50 tests passed, 16.049 s; no skips |
| Six generated demonstration streams | Every named output exactly matched the independent interpreter |
| New HEVC/AV1 hardware execution | Not tested |

The 50-test suite consists of 18 HEVC tests, 12 AV1 tests, 12 semantic/H.264
adapter tests, and eight cross-target tests. Times are test-suite durations,
not decoder-performance measurements.

```sh
python3 -m codec_compute.av1.bootstrap
python3 -m unittest discover -s codec_compute/tests -v
python3 -m codec_compute.demo

# Existing compiler baseline, from the repository root:
PYTHONPATH="$PWD:$PWD/05-h264-compiler" python3 -m unittest -v \
  test_compiler test_native_ops test_interpolation test_schedule
```

The historical H.264 baseline exercised available FFmpeg, OpenH264, and VA-API
decoder paths. New shared H.264 lowering was validated through FFmpeg. HEVC
uses FFmpeg's software decoder. AV1 native tests and demos run both libdav1d
and libaom-av1 through FFmpeg. NVDEC is unavailable on this host.

## Exactness and actual decoder work

- Averaging covers all 65,536 byte pairs on all three targets. The cross-target
  suite uses 16 chunks of 64x64 pixels to respect the existing H.264 writer's
  level-3.1 PCM access-unit bound. AV1 and HEVC native tests also use 256x256.
- HEVC saturating addition and subtraction each cover all 65,536 pairs in
  standalone instructions and resident programs. Thresholds cover every byte
  and every cutoff from -1 through 257. Signed weight and denominator boundaries
  exercise rounding before clipping.
- AV1 copies, positive and negative translations, compound averaging, repeated
  sources, odd sums, random images, and extrema agree with independent formulas
  through two decoders. All eight reference slots and repeated slot replacement
  are tested. Header inspection checks fixed entropy state, no inherited
  primary context, lossless quantization, and precise refresh masks.
- Changed input pixels retain identical computation tails. AV1 input-only
  serialization plus the neutral-compiled tail exactly matches a complete
  controlled encode with those same real inputs. Serializer spies verify that
  runtime receives inputs only. HEVC spies verify its sole upload call contains
  the original inputs and the encoder schedule's duplicate first input.
- Chains remain inside decoder reference buffers: HEVC tests retain an old
  input across 128 stages; AV1 tests retain and expose 20 composed averages.
  Outputs remain observable after reference slots are released.
- Shared tests replace the interpreter with a function that fails if called
  during encoding or decoding. Raw forged IR bounds fail compilation before
  they can incorrectly influence instruction selection.

## Demonstration artifacts

`python3 -m codec_compute.demo` generates the following at 32x32 logical pixels:

| Program | Target | Input pictures | Compute pictures | Prediction bytes |
| --- | --- | ---: | ---: | ---: |
| Two dependent rounded averages | H.264 | 2 | 2 | 343 |
| Two dependent rounded averages | HEVC | 2 | 2 | 34 |
| Two dependent rounded averages | AV1 | 2 | 2 | 60 |
| Saturating add → subtract → threshold | H.264 | 2 | 4 | 631 |
| Saturating add → subtract → threshold | HEVC | 2 | 4 | 71 |
| Average → clamp translation → average | AV1 | 2 | 3 | 92 |

These are serialization sizes, not speed measurements. H.264 includes guarded
I420 pictures; HEVC has different headers and geometry; AV1 uses monochrome IVF
and its byte total includes packet framing. Input-loading work and all emitted
bytes also matter when comparing complete invocations.

The two-average output SHA-256 is identical for all three codecs:

```text
4cefad337cdb8515e13a14b0de16f0a5f6b7ae67be98f8d73fb6e1dd1162be43
```

Generated streams, per-program manifests, and `validation.json` are available
under ignored `codec_compute/artifacts/` and can be regenerated by the demo.
The report records output hashes, prediction hashes, decoder names, and counts.

These results establish the implemented subset on the tested decoders. They do
not establish general AV1 affine/threshold synthesis, hardware portability,
cross-target runtime speedups, or a complete port of the original applications.
