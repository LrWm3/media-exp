# CPU execution versus decoder computation

**No elapsed-time win was found. One narrow host-CPU win survives CPU tuning:**
the 512×512/64-stage synthetic recurrence uses about 14.2% less measured process
CPU on NVDEC, but finishes 6.22× slower. Every other tested case loses both
metrics. No useful-application throughput or energy advantage is demonstrated.

This experiment asks whether decoder computation offers an advantage in elapsed
time or host CPU consumption. It compares identical integer outputs on the same
machine, using the shared IR compiler and an independent optimized C++ CPU
implementation. Energy is outside the measured scope: accessible interfaces did
not provide usable power or energy readings under the available account.

## Alternatives tested

| Workload | Compute pictures | What it tests |
| --- | ---: | --- |
| Copy control | 0 | Input/upload/readback overhead; CPU performs a real copy. |
| Rounded average | 1 | Decoder-native arithmetic with almost no instruction depth. |
| Patch atlas | 1 | Integer block routing into a quarter-area ROI; other pixels are zero. |
| Fractional warp/blend | 1 | Two independent block fields and their average fused into one prediction. |
| Resident recurrence, 16 rounds | 16 | One uploaded image, repeated fractional warp/blend, one final readback. |
| Resident recurrence, 64 rounds | 64 | More computation per upload while intermediates stay inside the decoder. |

All six run at 128×128, 256×256 and 512×512 active pixels. The larger
follow-up uses explicit High level 5.1 resource limits. The recurrence is a stress
case for retained intermediates, not evidence of a new useful application.
Both backends execute every rounded/clipped iteration; the CPU keeps scratch
inside one native call and returns only the final image.

Two additional runtime alternatives address known overheads:

- Native I_PCM serialization replaces Python's per-byte input loop. The
  returned access units must match the original Writer exactly and still pass
  its resource limits. Prepared-input decoding is measured separately.
- Finite batches of 64 jobs test throughput without a completion fence after
  each job. They retain independent input uploads, end at one EOS, and include
  file I/O, process/pipeline startup and shutdown. These batch figures are not
  online request latency. Batches cover the three one-picture tasks at 256×256 and 512×512.

## Matched semantics and timing

CPU kernels are single-threaded C++17 compiled with `-O3`, using reusable scratch
and fixed block plans. The fractional sampler specializes each of the sixteen
H.264 phases, uses padded zero boundaries, and applies separable six-tap filters
with signed intermediate precision. Compiler vectorization diagnostics and
build flags are retained. This is an optimized portable baseline, not a claim
to the fastest architecture-specific CPU implementation.

The 512×512 follow-up also includes a stronger recurrence CPU baseline. Its
fixed pair of fractional fields shares the horizontal six-tap calculations,
uses a six-row signed ring and retained row results, and avoids filtering the
same half-samples twice. Every round and both rounded averages remain exact;
filtered values just outside the active rectangle are retained. It uses no
handwritten SIMD or pixel-dependent shortcut. Both generic and specialized CPU
results are reported, and the stronger baseline governs the comparison.

The CPU produces a compact byte image; NVDEC necessarily handles guarded I420
surfaces. Both start from the same compact host inputs in the native-upload
comparison and finish with host-readable exact outputs. Every timed request
checks equality. CPU calls include ctypes dispatch, output-byte allocation and
comparison. Decoder requests include serialization where applicable, protocol
checks, IPC, decoding, readback and comparison. Required request allocations
remain timed; context setup, reusable scratch, compilation and loading do not.

Eight fixtures include structured scenes, noise, zero and 255. Short tasks cycle
through them for 64 requests per trial; chains use eight requests. Five repeats
rotate task/mode order. This is warm-cache behavior with fixed motion fields;
there is no pixel-result memoization. Resident decoder creation, one complete
warmup and final close are excluded from warm timings and reported separately.
Every online job uses one additional decoded IDR fence. Batched jobs omit it.

Host CPU time is measured independently from elapsed time. The parent process
clock includes native CPU work, encoding, validation and diagnostic threads.
For a live decoder worker, `clock_getcpuclockid` obtains its whole-process CPU
clock; `/proc` supplies a coarser user/system split. Finite batches use reaped
child usage. These counts exclude other processes, interrupts and kernel
workqueues, and do not measure energy. Multiple active threads can consume more
CPU seconds than wall seconds. In particular, `RUSAGE_CHILDREN` is not used to
estimate CPU consumed by a still-running worker.

Before timing, all CPU results and native input bytes are checked, then every
real decoded picture is compared through live NVDEC sessions. Local exports
also compare full I420 through FFmpeg, OpenH264 and VA-API. Small integration
tests compare every workload, including 64-round recurrence, against the
semantic IR and FFmpeg. Independent sampling tests exercise all fractional
phases, signed coordinates, boundary conditions and full-byte values.

## First result: 128×128 and 256×256

**No speed or host-CPU advantage was observed.** Optimized CPU execution wins
both metrics in all twelve cases, against every measured decoder mode. Even the
slowest CPU trial is faster and uses less process CPU time than the best decoder
trial for that case. This is a negative result for this implementation and input
regime, not a statistical proof about every decoder or workload.

The [complete report](assets/results.json), pinned to source `ed20a71`, contains
270 timing trials and five repeats per mode. Representative 256×256 medians are
milliseconds per job. Both host-input paths start with the same compact byte
images; NVDEC uses the native uploader, not the earlier Python input encoder.

| Task | CPU elapsed | NVDEC elapsed | CPU process time | NVDEC process time |
| --- | ---: | ---: | ---: | ---: |
| Rounded average | 0.107 | 13.466 | 0.103 | 7.541 |
| Patch atlas | 0.042 | 9.862 | 0.042 | 5.420 |
| Fractional warp/blend | 0.546 | 13.014 | 0.542 | 7.783 |
| 16-stage recurrence | 9.951 | 72.965 | 9.929 | 16.517 |
| 64-stage recurrence | 39.539 | 269.808 | 39.434 | 57.125 |

Native serialization of the two 256×256 interpolation inputs takes **0.571 ms**
in its separate mode. Prepared-input NVDEC interpolation still takes **12.784 ms**
and **7.037 ms process CPU**, versus **0.546/0.542 ms** for native CPU execution.
The missing speedup therefore cannot be explained solely by the original slow
Python uploader. These separately measured medians must not be subtracted to
estimate decoder-only time.

Finite64 native-input batches measure **14.328 ms/job** for average,
**10.960 ms/job** for atlas, and **14.410 ms/job** for warp/blend, including their
startup/file-I/O/shutdown costs. They do not reveal a throughput advantage.
The 64-stage chain removes repeated uploads and intermediate readback, but at
256×256 still consumes **57.125 ms** process CPU versus **39.434 ms** for the CPU.
The decoder's process overhead remains significant even though the arithmetic
executes on a separate engine.

All **96 hardware correctness jobs** pass: **1,552 decoded pictures including
96 fences**, with every real picture checked (**1,456 I420 readbacks,
118,530,048 bytes**). The local export passes **288 decoder checks**.
The [full index](assets/full-bundle.json) binds the report to that corpus;
[representative fixtures](assets/bundle.json) retain one job per case.
The [128×128 contact sheet](assets/reference-outputs-128.png) and
[256×256 contact sheet](assets/reference-outputs-256.png) show saved decoded-reference
pixels; the passing hardware report verifies those same pixels exactly.

## Larger follow-up: 512×512

The [complete larger report](assets/large/results.json), pinned to `6eb0cc2`,
contains 160 timing trials. All 48 hardware correctness jobs pass: 776 decoded
pictures including 48 fences, with all 728 real pictures checked byte-for-byte
(323,162,112 I420 bytes). Local export passes 144 decoder checks. The
[full index](assets/large/full-bundle.json), [representative fixtures](assets/large/bundle.json)
and [decoded-reference capture](assets/large/reference-outputs-512.png) retain
the evidence and exact subset provenance.

The shared-filter CPU implementation changes the interpretation substantially.
For the 16-stage chain, the generic CPU takes 39.719 ms process CPU, while the
specialized CPU takes 20.637 ms. NVDEC's 25.673 ms therefore loses to the stronger
baseline even though it beats the generic implementation.

For the 64-stage chain, a narrow process-CPU advantage remains in this run:
NVDEC consumes **70.078 ms**, versus **82.050 ms** for the shared-filter CPU,
a **14.6% reduction**. The five trial ranges do not overlap: 69.343–71.047 ms
versus 82.021–82.164 ms. However, elapsed time is **509.032 ms versus 82.185 ms**,
so NVDEC finishes **6.19× slower**. This is a synthetic recurrence, not a
useful application speedup or an energy measurement.

| Task | CPU elapsed | NVDEC elapsed | CPU process time | NVDEC process time |
| --- | ---: | ---: | ---: | ---: |
| Rounded average | 0.394 | 27.891 | 0.391 | 17.732 |
| Patch atlas | 0.155 | 20.179 | 0.153 | 12.458 |
| Fractional warp/blend | 2.115 | 27.983 | 2.107 | 17.057 |
| 16-stage recurrence, shared-filter CPU | 20.677 | 138.657 | 20.637 | 25.673 |
| 64-stage recurrence, shared-filter CPU | 82.185 | 509.032 | 82.050 | 70.078 |

Values are milliseconds per job, five-repeat medians, native host uploads for
NVDEC. Finite64 native batches improve 512×512 warp/blend throughput to
24.249 ms/job, but still lose to the CPU's 2.115 ms/job. Prepared-input batch
warp/blend takes 19.954 ms/job. Neither mode establishes a speed advantage.

## CPU compiler-flag sensitivity

The [CPU-only follow-up](assets/large/cpu-tuning.json), produced by
[tune_cpu.py](tune_cpu.py) at source `9302379`, rebuilds the frozen `6eb0cc2`
generic and shared-filter kernels with three flag sets: portable `-O3`,
`-O3 -mcpu=native`, and native plus `-funroll-loops`. All six builds pass exact
output checks. The runner records flags, compiler/source/script/binary hashes,
rotated order and 60 additional timing trials; all 96 initial, 60 warmup and
480 timed calls match their saved outputs.

Portable controls closely reproduce the main run: the 64-stage shared-filter
CPU takes 82.141 ms process CPU versus the earlier 82.050 ms. The best tested
variant, shared filters with native code generation and loop unrolling, takes
**81.703 ms process CPU and 81.840 ms elapsed**. Its CPU range is
81.599–81.833 ms, still above NVDEC's 69.343–71.047 ms. Thus the measured native
NVDEC path uses **14.2% less process CPU**, while completing **6.22× slower**.
The best 16-stage variant takes 20.576 ms CPU / 20.623 ms elapsed, retaining
its advantage over NVDEC on both metrics.

This is stronger evidence than comparing against an unoptimized interpreter
or the generic kernel alone. It remains a bounded implementation comparison:
there is no claim of an optimal CPU kernel, statistical generality, energy
savings, or improved concurrent application throughput. A newer compiler or
handwritten SIMD might reduce or erase the small process-CPU gap.

## What could change the conclusion

The practical conclusion is negative so far: no elapsed-time, application
throughput or energy win has been demonstrated. The narrow process-CPU result
concerns this implementation, fixed fields, these image sizes and
host-input/host-output requests. It does not prove that decoder computation can
never be useful. Plausible changes would need new measurements:

- Already-decoded video as the actual input, with results consumed on the GPU,
  could eliminate input-picture serialization, upload and host readback. The
  prepared-input mode here still decodes fresh input pictures; it is not that
  integration.
- A lower-overhead submission/driver path could reduce per-picture process CPU
  and latency. There is a concrete syntax opportunity: the 512×512 writer emits
  98 slices per prediction because each active row separates both zero guards
  from its interior. For these uniform zero-preserving averages, reusing the
  same prediction weights with zero guard motion could reduce this to 34 row
  slices. This has not been implemented or timed; the current helper already
  submits each whole access unit as one buffer, so it would reduce syntax and
  parsing rather than application submissions. Native input serialization alone,
  finite batching and retained intermediates were already tested here; none should be presented as an
  untested solution to the negative smaller-surface result.
- A useful workload with a long resident computation graph and enough latency
  tolerance could make the observed process-CPU savings worthwhile. The
  64-stage recurrence demonstrates that tradeoff; it does not establish a
  useful application. Its stronger CPU baseline already shares overlapping
  filter calculations, so that optimization must remain in future comparisons.
- Energy or performance under competing CPU/GPU work could differ from isolated
  timings. Neither lower energy nor improved application throughput follows from
  having a separate engine. Accessible telemetry did not permit an energy test.

## Reproduce

From the repository root:

```sh
python3 codec_compute/workloads/cpu_comparison/build.py \
  codec_compute/artifacts/cpu-comparison-build
python3 -m codec_compute.workloads.cpu_comparison.prepare \
  --cpu-helper codec_compute/artifacts/cpu-comparison-build/libcpu.so \
  --output codec_compute/artifacts/cpu-comparison --sizes 128 256 --jobs 8
python3 codec_compute/workloads/cpu_comparison/benchmark.py \
  codec_compute/artifacts/cpu-comparison \
  --cpu-helper codec_compute/artifacts/cpu-comparison-build/libcpu.so \
  --pcm-helper codec_compute/artifacts/cpu-comparison-build/libpcm.so \
  --planes-helper codec_compute/artifacts/cpu-comparison-build/libcpu_planes.so \
  --build-info codec_compute/artifacts/cpu-comparison-build/build.json \
  --nvdec-helper /path/to/nvdec_decode --repeats 5 --requests 64 \
  --output codec_compute/artifacts/cpu-comparison/results.json
```

For the larger follow-up, export with `--sizes 512` into a separate bundle
directory, then benchmark that directory with the same options. The current
runner includes the stronger CPU recurrence baseline; the archived initial
report predates it and preserves its original source revision.

Build the CPU and PCM libraries on the measurement host. The portable runner
requires Python 3.8+, the frozen bundle, relevant runtime sources with preserved
relative paths, and the existing NVDEC helper. It imports no compiler or model.
The report checks source, bundle and helper hashes, and records compiler flags
and raw per-trial wall/CPU measurements. No clocks or power settings are changed.

Run the CPU flag follow-up on the measurement host after decoder timing finishes:

```sh
python3 codec_compute/workloads/cpu_comparison/tune_cpu.py \
  codec_compute/artifacts/cpu-comparison-large \
  --output codec_compute/artifacts/cpu-comparison-large/cpu-tuning.json \
  --build-dir codec_compute/artifacts/cpu-comparison-tuning-build --repeats 5
```

It selects `-mcpu=native` on AArch64 or `-march=native` on x86, records all build
diagnostics, and fails explicitly if a requested variant cannot be built.
No compiler or device settings are installed or changed.
