# Validation

Recorded locally on 2026-10-23 with Python 3.12.13, PyTorch 2.5.1+cpu,
NumPy 2.2.6, FFmpeg 6.1.1, and GStreamer 1.24.2. Intel hardware decoding used
`vah264dec` through `/dev/dri/renderD128`.

## Official MNIST test split

```sh
python3 run.py --samples 0 --verify --output out/test-set.h264
```

All 1,984 original test images labeled 3 or 8 were decoded in original dataset
order, after freezing the model. The test split was excluded from training,
architecture choice, and checkpoint selection.

| Ground truth | Predicted 3 | Predicted 8 | Total |
| --- | ---: | ---: | ---: |
| 3 | 952 | 58 | 1,010 |
| 8 | 42 | 932 | 974 |
| **Total** | **994** | **990** | **1,984** |

**Accuracy: 1,884 / 1,984 = 94.9597%.** This is a two-class result, not
ten-class MNIST accuracy. There are 100 recognition errors despite exact
execution of the model.

Every pooled input bit, every hidden activation, and every classification bit
matched independent Python integer inference. Original PyTorch checkpoint
hidden activations and classifications also matched all 1,984 decoded results.
There were zero decoder/model mismatches. Seed pixels, picture counts, neutral
chroma, and permanent zero guards were checked for every decoded batch.

The [test report](assets/test-set.json), [all predictions](assets/predictions.csv),
and [checkpoint comparison](assets/checkpoint-parity.json) are checked in.
The CSV records original test indices; the report records stream/model/raw-output
hashes. The large test stream is reproducible with the command above and is not committed.

## Runtime and stream size

The test used 248 batches of eight digits, each taking 1,802 decoded pictures:
446,896 pictures in total. The stream occupied 516,791,756 bytes. Each batch
includes 210 exact reference copies; maximum reference age was 16 pictures.
The largest spatial offset was 26 pixels.

| Work | Recorded time |
| --- | ---: |
| Compile reusable prediction instructions | 2.960 s |
| Encode all input seeds | 5.986 s |
| FFmpeg decoding and raw-picture readback | 55.239 s |
| Extract/check decoded results and reference inference | 16.588 s |
| Full batch loop, including I/O and hashing | 83.191 s |

Decode/readback throughput was approximately 35.9 digits/second; the complete
batch loop processed approximately 23.8 digits/second. These are single local
observations, not a controlled benchmark. Training, initial dataset download,
and capture rendering are excluded. The stream declares 30 pictures/second
for playback; the runner decodes without real-time pacing.

## Hardware and independent decoders

```sh
python3 run.py --decoder vaapi --verify --preview --output out/demo.h264
```

The capture uses the first 16 eligible test digits without selection by
prediction. It shows original/binarized input, actual decoded pooled pixels,
actual decoded hidden bits, and the class read from the output pixel.
Labels, class-name text, and enlargement are presentation added afterward.
The [hardware demo report](assets/demo.json) records its outputs and hashes;
the [stream](assets/mnist.h264), [GIF](assets/decoder-capture.gif), and
[PNG](assets/decoder-capture.png) accompany it.

The automated suite compares eight test digits' full pooling/hidden/output
results across FFmpeg, OpenH264, and Intel VA-API. The full 1,984-image accuracy
run uses FFmpeg; it is not presented as a full hardware test-set evaluation.

## Automated checks

```sh
.venv/bin/python -m unittest -v
```

**11 tests passed in 21.798 seconds; none skipped** in the recorded environment.
The suite covers:

- Actual inference through all three decoder backends.
- All 16 binary 2×2 patterns at each pooling threshold from 1 through 4.
- Signed weights, extreme hidden thresholds, split biases/threshold offsets,
  constant inputs, and the farthest pooled input at offset (26,26).
- One-neuron output extremes and 16-neuron reference schedules.
- Swapped input tiles, partial batches, and byte-identical prediction tails.
- Encoding while the host reference-model function is disabled.
- Original trained PyTorch checkpoint parity.
- Declared motion bounds, training/validation separation, corrupt dataset
  checksums, malformed inputs, and missing decoded results.

An independent arithmetic review checked 15,290,746 combinations of negative
weight count, hidden threshold, and unsigned accumulated score, with no
threshold mismatch, plus 784 legal initial-bias cases. These are arithmetic
checks, not millions of separate decoder runs.

The shared writer gained configurable motion bounds and validation. Regression
suites for experiments 01, 02, and 03 passed **9, 11, and 11 tests** respectively,
with none skipped. Total automated tests across the four experiments: **42**.

## Reproducibility

```sh
.venv/bin/python train.py --output out/reproduced.json
cmp model/weights.json out/reproduced.json
```

Repeated training reproduced frozen weights byte-for-byte. The selected model
SHA-256 is `260525f39a96214890e0044e02a8ba46fdb15865ffdc54ba5b214fe846e45eb7`.
The checkpoint, training history, split-index hashes, and validation selection
records are supplied under `model/`.
