# Static-field resampling experiments

**Motion-compensated interpolation is the strongest architectural fit here.**
Two independently warped inputs and their rounded average fit into one H.264
prediction picture. Patch routing and bandwise correction each take one picture
as well. Compensated residuals, dense polarity maps, and stereo candidate costs
execute exactly, with progressively longer arithmetic chains.

These are experiments with **externally supplied static motion fields**. They do
not estimate optical flow, solve occlusions, model a physical rolling shutter, or
produce asynchronous/logarithmic sensor events. The compiler specializes each
field into a cached prediction tail; changing input pixels does not change that
tail. Updating a field requires another compile/export.

## Tasks and exact costs

All tasks operate on **128×128 byte images**, with **160×160 coded I420 surfaces**.
The table counts computation pictures; upload pictures are listed separately.
Baseline materializes spatial reads; fused mode moves eligible reads directly
into their scalar consumers, independently for each source of a B prediction.

| Task | Inputs | Baseline → fused compute pictures | Fused prediction bytes | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Patch atlas | 1 | 1 → 1 | 587 | Route selected 16×16 tiles into a 64×64 ROI; zero the remainder. |
| Motion interpolation | 2 | 3 → 1 | 771 | Average two independently sampled fractional block fields. |
| Bandwise correction | 1 | 1 → 1 | 505 | Apply one integer horizontal offset per 16-row band. |
| Compensated absolute residual | 2 | 4 → 3 | 2,094 | Absolute difference after a fractional reference warp. |
| Dense polarity maps | 2 | 7 → 6 | 3,492 | Two masks: current−warped-reference ≥20 and reverse difference ≥20. |
| Stereo candidate costs | 2 | 12 → 9 | 5,901 | Three absolute difference maps, without aggregation or disparity selection. |
| Passthrough control | 1 | 0 | 0 | Return the uploaded input picture. |
| Unwarped average control | 2 | 1 | 642 | Average the exact same two inputs used by motion interpolation. |

Patch routing changes pixel placement within the existing surface. Its 64×64
active ROI **does not reduce coded geometry, decode work to a 64×64 surface, or
readback allocation**. Each selected 160×160 I420 picture still returns 38,400
bytes. The single-input passthrough control has its own fixture at this geometry;
it supplies coarse startup/upload context, not a matched subtraction for atlas
or rolling-shutter timing. The average control is a matched two-input baseline.

The stereo candidates use **reference read offsets** of −0.5, +0.5, and +1.5
pixels: an output pixel at `x` reads the reference at `x + offset`. They are dense
cost maps, not a depth or correspondence solution. The polarity maps are dense
frame-difference thresholds on literal byte brightness, with no event timestamps
or logarithmic response.

## Codec-defined sampling and quality witness

`media.gather(image, offsets, block_size=16, border="zero")` expresses integer
block reads. `media.sample_h264(image, qx, qy)` and `media.gather_h264(image,
offsets, block_size=16)` deliberately name H.264's quarter-pixel luma filter.
Those fractional operations use its six-tap filtering, specified rounding and
clipping; they do not promise another codec's or a generic bilinear filter's
pixels. Motion offsets are relative source reads, in pixel or quarter-pixel
units respectively. This experiment uses H.264 only.

The main interpolation case supplies two different fractional fields, including
different horizontal and vertical phases per block. Both inputs keep distinct
fields when fused into a single prediction. Image boundaries extend with zero
before filtering, and the resulting edge samples are checked exactly.

A separate controlled witness uses a nonlinear analytic sinusoidal texture
translated by **±2.5 pixels**. It decodes both an ordinary average and a compensated
average using the known half-pixel offsets, and compares them with the analytic
midpoint on an interior **112×112 crop**. That crop excludes displacement and
filter support at image edges. The witness demonstrates the value of supplied
motion on this texture; it is not evidence about estimated flow, real video
quality, disocclusions, or a broad image distribution. The capture labels the
analytic midpoint separately from actual decoded outputs.

## Validation and benchmark protocol

Preparation uses eight changing fixtures per task: fixed scenes and deterministic
full-byte noise, with static fields shared by every fixture. Each named output
must match both the semantic IR and independent task formulas. Fractional task
references use the original compiler's separately tested normative interpolation
helper; the shared IR uses its own sampling module. Saved streams are decoded by
FFmpeg, OpenH264, and available VA-API. The exporter checks complete I420 equality
across decoders, all named outputs, zero guard luma, and neutral chroma, then saves
the actual decoded planes and their hashes.

The portable runner compares against those **frozen independently validated
references**; it does not rerun compiler synthesis or fractional formulas. It
validates every baseline/fused stream, both controls, and the known-motion witness
before timing, including untimed variants. It also recomputes witness errors from
actual returned pixels and the saved analytic midpoint. Capture generation
requires passing decoder evidence and binds every displayed input/output plane
to its exact crop in the saved full I420 frame. Binary masks display 0/1 as
black/white; other planes retain literal byte intensity.

Timing covers all fused tasks, both controls, and the unfused interpolation
baseline, with batches of **one and eight independent jobs**. Each case/batch/mode
gets one warmup and five measured repeats, with rotating order. Reports preserve
all samples and medians. Four modes remain separate:

| Mode | Included work |
| --- | --- |
| Input serialization | Validate and serialize already loaded input boards into I_PCM pictures; compare each prefix byte with its prepared reference. |
| Pre-encoded, all pictures | Batch assembly and file I/O, process/pipeline startup, parsing, decoding, all surface copies, diagnostics, and exact returned-I420/named-output checks. |
| Pre-encoded, named outputs | Same scope with deduplicated output-picture selection. NVDEC skips unselected surface copies; FFmpeg copies all pictures before selecting in Python. |
| Host inputs to named outputs | Measured directly: serialize/check inputs, append the cached tail, write streams, then run the selected-output decode path. |

Compilation, motion-field setup, fixture/reference generation and loading,
source/hash auditing, initial guard checks, network transfer, and report writing
are outside timing. No cache-flushing protocol is used. Every decode call opens
a new session and runs a finite batch to EOS. Per-job figures are amortized
throughput, not individual latency or steady-state decoder throughput. No GPU
arithmetic baseline, concurrent GPU workload, or energy result is claimed.
FFmpeg and NVDEC run in separate environments, so their absolute timings cannot
isolate hardware performance.

The uploader explicitly probes 256×256 zero and byte-ramp inputs. Both are rejected
by the existing High level 3.1 initial I_PCM access-unit limit. No large-image
support is inferred from the semantic graph's ability to represent those shapes.
Input escaping makes some near-limit shapes content dependent; the measured
128×128 canvas accepts the complete byte fixtures here.

## Reproduce

From the repository root:

```sh
python3 -m unittest codec_compute.tests.test_resampling_workloads -v
python3 -m codec_compute.workloads.resampling.demo \
  --output codec_compute/artifacts/resampling --size 128 --jobs 8
python3 -m codec_compute.workloads.resampling.capture \
  codec_compute/artifacts/resampling \
  --output codec_compute/artifacts/resampling/capture.png
python3 codec_compute/workloads/resampling/benchmark.py \
  codec_compute/artifacts/resampling --backend ffmpeg \
  --batch-sizes 1 8 --repeats 5 \
  --output codec_compute/artifacts/resampling/ffmpeg.json
```

For a short correctness export, use `--size 32 --jobs 2`; its runner must use
`--batch-sizes 1 2`. The default export prepares eight jobs to match benchmark
defaults. The complete generated corpus stays under ignored `codec_compute/artifacts/`.

The NVDEC host needs the prepared bundle, this directory's `benchmark.py`,
`codec_compute/workloads/temporal_benchmark.py`, and
`decoder_core/{__init__,decode,bitstream}.py` with their relative layout preserved.
The runner uses Python 3.8 and the standard library, plus the existing decoder
helper. The identical Writer source loads with postponed annotations and every
serialized input prefix is checked byte for byte; no compiler installation is
required on that host.

```sh
python3 codec_compute/workloads/resampling/benchmark.py \
  codec_compute/artifacts/resampling --backend nvdec \
  --helper /path/to/nvdec_decode --batch-sizes 1 8 --repeats 5 \
  --output nvdec.json
```

`--cases` can select exact bundle case keys for timing; full validation still
covers all prepared streams. `suite.json` additionally supports the existing
`05-h264-compiler/optimizations/ir/validate.py` portable validator.
