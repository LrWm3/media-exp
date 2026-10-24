# Temporal workload benchmarks

This benchmark separates host input serialization, decoding saved streams, and
an actual host-input-to-output run. All four tasks use the shared IR and H.264:
positive change, absolute change with a cutoff-24 mask, two-frame minimum/maximum
and range, and half-up rounded temporal averaging. It measures independent
standalone jobs, each uploading both inputs. It does not implement a persistent
camera stream or reuse input surfaces across jobs.

The default fixture is **128×128 logical pixels**, inside **160×160 coded I420
pictures** with a 16-pixel zero guard. Sixteen distinct deterministic input pairs
include scene motion, full-byte noise, equal pixels, extrema, threshold boundaries,
and odd sums. Every task receives the same pairs. The existing writer rejects
256×256 and 160×160 logical I_PCM inputs at its High level 3.1 initial access-unit
size limit; this benchmark keeps that limit intact.

## Measured results

The saved [FFmpeg](assets/ffmpeg.json) and [NVDEC](assets/nvdec.json) runs both
pass exact validation of **64 streams, 320 pictures, 12,288,000 I420 bytes, and
112 named outputs containing 1,835,008 pixels** before timing. Every measured
trial also passes exact returned-I420 and named-output comparisons. The full
shared-IR test suite passes **133 tests**, including four benchmark protocol
checks; the portable runner was additionally exercised on Python 3.8.

Median elapsed milliseconds for **sixteen independent jobs per batch**:

| Backend | Task | Input serialization | Saved streams, all pictures | Saved streams, named outputs | Host inputs to named outputs |
| --- | --- | ---: | ---: | ---: | ---: |
| FFmpeg | Positive change | 51.88 | 40.68 | 50.14 | 92.80 |
| FFmpeg | Absolute change + mask | 52.74 | 42.49 | 47.38 | 97.97 |
| FFmpeg | Temporal envelope | 54.66 | 44.71 | 43.03 | 101.72 |
| FFmpeg | Temporal average | 57.24 | 41.47 | 37.08 | 90.88 |
| NVDEC | Positive change | 318.82 | 375.11 | 377.39 | 710.29 |
| NVDEC | Absolute change + mask | 320.24 | 529.49 | 517.54 | 840.47 |
| NVDEC | Temporal envelope | 311.54 | 543.00 | 539.09 | 864.41 |
| NVDEC | Temporal average | 323.91 | 377.97 | 370.20 | 698.09 |

The backends ran in different environments and Python versions, so this table
does not isolate hardware decode performance. All modes include the checks
described below. **Input loading and startup are substantial costs**: on the
NVDEC environment, serializing sixteen input pairs alone took 312–324 ms. A
single-job saved-stream run with named-output readback took 265–283 ms, versus
370–539 ms for sixteen jobs. Those values include process/pipeline startup,
I/O, parsing, decoding, copies, and exact checks.

Selected NVDEC readback reduces copied pictures and bytes exactly, but did not
produce a consistent elapsed-time win in this finite-batch setup. FFmpeg's
selected mode still copies every decoded picture. These results establish the
current standalone-job costs; they do not establish an acceleration benefit.

Static costs per independent job, including the two input pictures:

| Task / first-job stream | Compute pictures | Total pictures | Named-output pictures | Prediction bytes | NVDEC I420 copied: all → outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Positive change](assets/positive-change/000.h264) | 1 | 3 | 1 | 644 | 115,200 → 38,400 B |
| [Absolute change + mask](assets/absolute-change/000.h264) | 5 | 7 | 2 | 2,942 | 268,800 → 76,800 B |
| [Temporal envelope](assets/temporal-envelope/000.h264) | 5 | 7 | 3 | 3,238 | 268,800 → 115,200 B |
| [Temporal average](assets/temporal-average/000.h264) | 1 | 3 | 1 | 642 | 115,200 → 38,400 B |

Each job's input prefix is **86,477–86,487 bytes**, shared byte for byte across
the four tasks for each input pair. Prediction tails are fixed across inputs.
Selected readback still decodes every picture and copies complete guarded I420
surfaces for the selected outputs, not just the active output pixels.

The [bundle index](assets/bundle.json) records source commit `d479dc3` and exact
artifact hashes. Both timing reports pin its SHA256
`272d7e69ff2f1114e1a40bb2cf78e937852db12daa0f8a6c40ef2710be25e95f`.
The committed evidence includes both full timing reports, the complete bundle
index, and each task's first-job stream and manifest. Regenerate the full ignored
bundle for all sixteen jobs and their input, output, and full-I420 files.

## Protocol

Each task is measured with finite batches of **one and sixteen jobs**, one warmup
per task/batch/mode, and five measured repeats. The task/batch/mode order rotates
between repetitions. Reports retain every sample, minimum, maximum, and median.

| Mode | Inside the measured interval |
| --- | --- |
| Input serialization | Validate and serialize two already loaded integer input boards per job as I_PCM; compare every resulting prefix byte with the frozen prefix. |
| Pre-encoded, all pictures | Assemble saved streams into one batch; file I/O, process/pipeline startup, parsing, decoding, readback, diagnostic validation, exact returned-I420 comparison, and named-output extraction/comparison. |
| Pre-encoded, named outputs | Same scope, requesting only the distinct pictures containing named outputs. NVDEC skips mapping/copying unselected surfaces; FFmpeg copies all pictures before Python selects the requested ones. |
| Host inputs to named outputs | Serialize both inputs, verify prefixes, append the cached computation tail, write streams, then perform the named-output decode mode. This is measured directly, not obtained by adding separate medians. |

Compilation, fixture/reference generation and loading, source/hash checks,
initial full padding/chroma validation, network transfer, and report writing
are outside the timing interval. Inputs and full references are resident in
Python memory before trials. No cache-flushing protocol is applied. Each decode
call opens a new process and decoder session and runs to EOS. A batch's elapsed
time divided by job count describes amortized throughput, **not single-job
latency or steady-state hardware decode latency**. No GPU arithmetic baseline,
concurrent GPU workload, or energy measurement is included.

Both input serialization and decoding run on each benchmark host. The portable
runner uses the unmodified `decoder_core/bitstream.py` source with postponed
type annotations, allowing Python 3.8 to load its `tuple[...]` annotations. The
source hash is recorded and every generated prefix must match the exporter
byte for byte. The compiler and its synthesis code are not imported there.

## Exact validation

Preparation checks each named result against both the semantic IR interpreter
and task formulas independent of lowering, then saves the actual FFmpeg output.
The portable runner independently recomputes every named task output from the
saved input planes before timing. It compares full actual I420 with the saved
reference for every job, including input pictures and intermediates, and checks
all guard luma for zero and all U/V samples for 128. Each timed run repeats exact
comparison of every returned I420 byte and every named output. Selected readback
therefore cannot silently omit a requested output or pass on checksums alone.

The bundle hashes all inputs, outputs, prefixes, streams, prediction tails,
manifests, full I420 references, relevant source files, and the portable runtime.
Reports pin the bundle hash and record actual decoder copy/readback counters.
The runtime refuses mismatched source hashes or artifacts.

## Reproduce

From the repository root:

```sh
python3 -m unittest codec_compute.tests.test_temporal_benchmark -v
python3 -m codec_compute.workloads.benchmark \
  --output codec_compute/artifacts/temporal-benchmark
python3 codec_compute/workloads/temporal_benchmark.py \
  codec_compute/artifacts/temporal-benchmark \
  --backend ffmpeg --batch-sizes 1 16 --repeats 5 \
  --output codec_compute/artifacts/temporal-benchmark/ffmpeg.json
```

Copy the prepared bundle, `codec_compute/workloads/temporal_benchmark.py`, and
`decoder_core/{__init__,decode,bitstream}.py` to the decoder environment, retaining
their relative layout. No compiler, training data, third-party Python package,
or automatic dependency installation is required on that host. Use the existing
NVDEC helper built by `decoder_core/build_nvdec.sh`:

```sh
python3 codec_compute/workloads/temporal_benchmark.py \
  codec_compute/artifacts/temporal-benchmark \
  --backend nvdec --helper /path/to/nvdec_decode \
  --batch-sizes 1 16 --repeats 5 \
  --output nvdec.json
```

Use `--width`, `--height`, and `--jobs` when preparing another geometry or job
count. Every requested batch size must fit within the prepared independent jobs.
The full generated bundle stays under ignored `codec_compute/artifacts/`.
