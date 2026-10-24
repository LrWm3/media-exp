# Validation

Recorded locally on 2026-10-23 with Python 3.12.13, PyTorch 2.5.1+cpu,
NumPy 2.2.6, FFmpeg 6.1.1, and GStreamer 1.24.2. Hardware decoding used
Intel VA-API through `vah264dec` and `/dev/dri/renderD128`.

## Decoder correctness

```sh
.venv/bin/python -m unittest -v
```

**11 tests passed in 3.428 seconds, none skipped.** Tests cover:

- All 512 possible binary 3×3 neighborhoods through FFmpeg, OpenH264, and
  Intel VA-API, comparing every learned feature bit and output bit.
- Signed weights, extreme thresholds, constant inputs, random borders, and
  zero padding against independent integer inference.
- One-channel and 16-channel networks, including reference refresh operations.
- Byte-identical prediction instructions after swapping the noisy input seed.
- Successful compilation with the host model's inference function disabled.
- Actual trained PyTorch checkpoint parity on eight held-out images.
- Quality improvement, disjoint training/test seeds, invalid inputs, and
  detection of missing verification results.

Verification also checks picture count, neutral chroma, permanent zero guards,
input seed pixels, and binary intermediate/output values. The full 256-image
held-out run below compared every decoded feature and output bit successfully.
Separately, original PyTorch checkpoint decisions matched the frozen network on
all 256 held-out images, with zero mismatches.

## Held-out quality and runtime

```sh
python3 run.py --samples 256 --seed 200000 --verify --output out/held-out.h264
```

The test set contains 256 binary 32×32 shape images: 262,144 pixels total.
Training, checkpoint selection, and testing use separate shape-seed ranges.

| Output | Pixel error |
| --- | ---: |
| Noisy input | 15.0715% |
| Trained CNN executed by FFmpeg's H.264 decoder | 3.2932% |
| Ordinary 3×3 majority filter | 3.3421% |

The CNN's advantage over the majority baseline is only 128 pixels on this set;
quality is comparable. This is a binary synthetic-shape experiment, not evidence
of general image-denoising performance.

The stream contains 25,088 decoded pictures and occupies 8,703,773 bytes.
Compilation took 1.559 seconds; FFmpeg decoding plus raw-picture readback took
1.058 seconds, about 242 input images/second for this batch. Timing includes
decoder startup, excludes verification and capture rendering, and is one local
measurement rather than a repeated benchmark. See [the full report](assets/held-out.json).

Each input uses 98 pictures: one seed, 92 calculation/display pictures, and five
reference copies. The maximum actual reference age is 16 pictures. The complete
schedule is in [program.json](assets/program.json).

## Hardware capture

```sh
python3 run.py --decoder vaapi --verify --preview
```

Eight new noisy images were decoded on Intel hardware with exact feature/output
agreement. Their 271,982-byte stream took 0.091 seconds to compile and 0.271
seconds to decode and read back. This smaller batch's timing is not directly
comparable to the 256-image software run.

On these eight demonstration images, errors were 15.1123% noisy, 2.9663% decoded,
and 2.9541% majority-filtered. The majority filter performs slightly better on
this demonstration batch. The [report](assets/demo.json), [executable stream](assets/denoising.h264),
[animated capture](assets/decoder-capture.gif), and [still capture](assets/decoder-capture.png)
are checked in. Captures use actual decoded output and feature pixels;
enlargement, labels, and the clean comparison image are added afterwards.

## Training reproducibility

The recorded 1,500-step CPU training run took 4.178 seconds. Checkpoint selection
used validation error, choosing step 1,150. Repeating training in the pinned
environment reproduced the frozen weights JSON byte-for-byte:

```sh
.venv/bin/python train.py --output out/reproduced.json
cmp model/weights.json out/reproduced.json
```

Frozen model SHA-256:
`ec8de200e9f598c3b6dcafd2503ab874620a2aecc56da2d146261263103ca32d`.
The JSON reports also record stream and decoded-raw hashes. The supplied model
can run without PyTorch; it is only needed for training and the checkpoint test.
