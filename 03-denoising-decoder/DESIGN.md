# A trained binary CNN inside H.264

The model has one binary input plane, eight binary feature planes, and one binary
output plane. Its 80 convolution/mix weights are learned signs, ±1. Nine learned
thresholds determine the activations. Inputs and activations use 0/1 encoding.
The effective receptive field is 3×3; there is no hidden external inference step.

## Training and freezing

`train.py` uses PyTorch Adam for 1,500 updates on procedurally generated binary
rectangles/ellipses, with fresh 15% independent bit-flip corruption per batch.
Weights use hard signs in the forward pass and a straight-through gradient.
Hidden activations use a hard threshold in the forward pass and a sigmoid
surrogate derivative. Latent weights are clipped after updates.

All forward feature values are binary during training. The final signed channel
sum, minus a learned threshold, is a logit for binary cross-entropy. Inference
thresholds that logit at zero. This is quantization-aware training, not training
a floating-point model and assuming it survives conversion.

There are 1,024 training shapes, 128 validation shapes, and 256 held-out test
shapes, using disjoint seed ranges. The selected checkpoint minimizes fixed
validation pixel error. The test split is evaluated after selection. Shapes and
noise are deterministic from the documented seeds.

Freezing takes each weight's sign and rounds each learned threshold upward.
For integer score `s`, `s >= real_threshold` is exactly equivalent to
`s >= ceil(real_threshold)`. Convolution thresholds are bounded to [−9,10];
mix thresholds to [−C,C+1], where `C` is the number of channels. Extreme thresholds
represent constant activations. The original checkpoint and frozen inference
were compared directly, with zero decision mismatches on the held-out split.

The quantization terminology follows [Larq's quantizer documentation](https://docs.larq.dev/larq/api/quantizers/).
This implementation uses PyTorch, not Larq. Convolution follows PyTorch's
[cross-correlation convention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html):
tap `(kx,ky)` reads offset `(kx−1,ky−1)`; kernels are not flipped.

## Exact decoder arithmetic

Let `C(x)` clip to [0,255]. P prediction computes a weighted, rounded shift and
offset. With denominator zero, B prediction computes:

```text
C(((w0*x + w1*y + 1) >> 1) + ((o0 + o1 + 1) >> 1))
```

Noisy input is seeded as 0/255 for visibility, then normalized to 0/1 by
`C(((x + 64) >> 7) - 1)`.

Each 3×3 channel takes ten pictures:

1. Initialize its accumulator with `9 + weight[0]*input[offset[0]]` using P prediction.
2. Add the other eight signed taps using B weights `(2, 2*weight[i])`.
3. On the last addition, set both B offsets to `55−threshold`. The value becomes
   `64 + signed_sum − threshold`.
4. Normalize using P weight 1 and denominator 7: `(value + 64) >> 7`.

The first nine pictures form the sum; the tenth produces its binary activation.
Before the last bias shift, partial sums stay within [0,18]. After it, every
possible value lies within [45,82]. Normalization therefore yields exactly 0 or
1, implementing the comparison at 64 without unintended clipping or rounding.

The 1×1 layer consumes each feature as soon as it is available. Initialize its
accumulator with bias `C` and add signed feature bits. The last addition applies
offset `64−C−output_threshold`, placing its score around 64. The same divide-by-128
operation implements the final threshold. The one-channel case folds this bias
directly into its initial P operation. Supported channel counts are 1–16.

Finally, two P pictures multiply the output bit by 127 and then by 3, clipping
the display result to 0/255. These display operations do not change the decision.

## Reference scheduling

The live registers are the normalized input, the current feature accumulator,
and the running 1×1 sum. After emitting a channel, its feature register is freed.
Saved feature frame numbers in the manifest are for inspection after decoding;
the compiler does not require them all to remain reference pictures.

Before each instruction, the scheduler checks whether any live reference's next
age reaches 16. It emits an exact P copy while that reference is still legal,
updates register aliases, and rechecks ages before resolving the instruction's
source frame numbers. The default network requires five such input copies.
All actual reads are at most 16 pictures old, including the refresh operations.

The eight-channel schedule is:

| Component | Pictures |
| --- | ---: |
| Noisy I_PCM seed | 1 |
| Input normalization | 1 |
| Eight 3×3 channels | 80 |
| Streaming 1×1 accumulation | 8 |
| Output threshold | 1 |
| Display scaling | 2 |
| Reference copies | 5 |
| **Total** | **98** |

Different inputs use byte-identical prediction instructions. Batch streams repeat
the same program after each new IDR seed, resetting references. Clean targets and
expected outputs are never stored in the bitstream.

## Padding and scope

The shared writer's permanent zero guard provides input zero-padding. Feature
planes also have zero guards, but the 1×1 layer never reads spatial offsets, so
hidden-padding values do not affect inference. Deblocking is disabled. No
fractional motion, residual coefficients, custom decoder, or host feedback is used.

This is a small binary denoiser for synthetic shapes, not RGB/grayscale photo
restoration. It can erase small structures and leave noise. The reported majority
baseline is essentially comparable; the small difference is not evidence of
statistical significance. Dimensions, slice rate, and reference-buffer bounds
are inherited from the shared High-profile level 3.1 writer. Tested hardware is
Intel VA-API; no general hardware compatibility or official conformance claim is made.
