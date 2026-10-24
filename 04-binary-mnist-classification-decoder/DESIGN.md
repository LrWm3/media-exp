# Binary handwritten-digit inference inside H.264

The network distinguishes handwritten **3s from 8s**. Its input is an original
28×28 MNIST image thresholded at intensity 128. The video decoder executes
2×2 binary max pooling, eight learned binary hidden neurons, and one learned
binary output neuron. A final decoded luma value of 0 means “3”; 255 means “8”.

The architecture is `28×28 binary → 14×14 pooled → 8 binary neurons → 1 bit`.
It has 1,576 learned ±1 weights and nine learned thresholds. The host loads and
binarizes inputs, writes the stream, and reads its result. There is no host-side
pooling, matrix multiplication, voting, or classification in the inference path.
Optional verification calculates an independent answer after decoding.

## Training and selection

`train.py` trains in PyTorch using hard binary weights and hidden activations
in every forward pass. Weight gradients use a straight-through estimator;
hidden activations use a sigmoid surrogate gradient. Adam uses learning rate
0.008, batches of 128, seed 29, and 6,000 updates. The final signed score minus
its learned threshold is divided by three for binary cross-entropy training;
this positive divisor does not change the inference decision.

The original MNIST training split contains 11,982 images labeled 3 or 8. A
deterministic stratified split reserves 500 of each digit for validation,
leaving 10,982 training images. The original test split remains separate.
Checkpoint and architecture selection use validation error only. Trials explored
7×7 versus 14×14 pooled features, 8 versus 16 hidden neurons, pooling thresholds,
and gradient settings. [Selection records](model/selection.json) document the
validation search; [training.json](model/training.json) records the final run.

The selected network retains 14×14 features using 2×2 pooling with `sum >= 1`,
equivalent to binary max pooling. Its best checkpoint is step 5,600, with 95.4%
validation accuracy. Extending the same training run to 12,000 updates did not
improve that checkpoint. Eight hidden neurons suffice for this experiment;
this is not a claim of optimal MNIST accuracy.

Freezing takes the sign of every weight and rounds learned thresholds upward.
For integer score `s`, `s >= real_threshold` is exactly equivalent to
`s >= ceil(real_threshold)`. The trained checkpoint and frozen integer model
are both supplied. PyTorch is unnecessary for compiled inference.

## Decoder arithmetic

Let `C(x)` clip an integer to [0,255]. Explicit weighted B prediction with
denominator zero calculates:

```text
C(((w0*x + w1*y + 1) >> 1) + ((o0 + o1 + 1) >> 1))
```

Using weights `(2,2)` and equal offsets `o` gives exact addition `C(x+y+o)`.
Using `(2,-2)` gives exact subtraction `C(x-y+o)`. Weighted P prediction supplies
scaling, offsets, and rounded shifts. No residual coefficients are present in
calculation pictures; deblocking is disabled.

### Input and pooling

Normalize visible seed pixels 0/255 to 0/1 using P weight 1, denominator 7,
and offset −1: `C(((x+64)>>7)-1)`.

For pooling, a B picture sums each input pixel and its right neighbor. Another
B picture sums that result and its lower neighbor, adding `64-Q`, where `Q` is
the pool threshold. A P picture with denominator 7 thresholds the 2×2 sum at
`Q`. For the trained model, `Q=1`. This takes three pictures and retains a dense
sliding pool plane; the classifier reads only its even-coordinate samples.

### Exact 196-input hidden neurons

For each hidden neuron let `N` be its number of negative weights. Its signed
score is `s = sum(w[i]*x[i])`, so `N+s` lies in [0,196] for **every** binary
input. Initializing the accumulator with bias `N` ensures all partial sums stay
in range, independent of tap order. Biases above 127 use two legal P offsets.

Gather the 196 pooled bits using motion offsets `(2*x,2*y)` for `x,y` in
[0,13]. Each subsequent B picture adds or subtracts one bit. The target
comparison `s >= T` is equivalent to `N+s >= K`, where `K=N+T`. Clamp `K` to
[0,197], preserving constant-true and constant-false cases outside the score range.

The last addition applies shift `192-K`. If this exceeds the legal offset 127,
split it across that B picture and an additional P picture. Intermediate
clipping is harmless here: every clipped high value already satisfies the
comparison, and every clipped low value fails it. Finish with:

```text
C(((C(N+s+192-K) + 64) >> 7) - 1)
```

For any 8-bit stored value, this yields exactly 0 below 192 and 1 at or above
192. It implements the threshold without overflowing an 8-bit accumulator.

### Output layer

Consume each hidden bit immediately into a signed output accumulator. With
`H<=16` neurons, bias `H` keeps partial sums in [0,2H]. The last addition adds
`64-H-output_threshold`; a denominator-7 P operation yields the final 0/1
classification. Two display predictions multiply by 127 and then by 3, clipping
to 0/255. Mapping that decoded bit to the class name requires no additional
learned calculation or decision.

## Scheduling and parallel inputs

Only three pictures need to remain live: pooled input, current hidden sum,
and output sum. Before a live reference reaches age 16, copy it with an exact
P prediction and update register aliases. The scheduler resolves operand frame
numbers after these copies. Every reference remains within the 16-picture DPB.

Each digit occupies a 32×32 tile: 28×28 input plus zero padding. Up to eight
tiles fit in a 128×64 working board, with a permanent 16-pixel outer guard.
Every digit's result sits at its tile's upper-left pixel. All reads contributing
to that result stay inside its own 28×28 input; unused pixels elsewhere may
contain other sliding-window evaluations. There is no exchange between digit
classifications.

The coded picture is 160×96. Identical prediction instructions execute on all
eight tiles in parallel. Larger datasets repeat the same instruction bytes
after each new IDR seed. Labels and expected predictions never enter the
compiler. The writer declares 7-bit motion-vector bounds for the 26-pixel
offsets, and rejects instructions exceeding its declared bounds.

| Stage | Pictures per batch |
| --- | ---: |
| I_PCM input seed | 1 |
| Input normalization | 1 |
| 2×2 pool sums and threshold | 3 |
| Eight hidden neurons, 197 pictures each | 1,576 |
| Output accumulation | 8 |
| Output threshold and display | 3 |
| Reference refresh copies | 210 |
| **Total** | **1,802** |

Only the seed changes with input. The trained weights determine the instruction
tail once; inference continues to work when the host reference-model function
is disabled. This is a finite compiled computation, without decoder-driven loops.

## Dataset provenance and limits

MNIST supplies grayscale 28×28 digits in separate training and test files.
See the [TensorFlow Datasets description](https://www.tensorflow.org/datasets/catalog/mnist)
and [Torchvision loader source](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
`data.py` uses the same original IDX resource filenames and MD5 checksums as
Torchvision, downloading from its HTTPS mirror. Cached files are checked before
use. The small demo fixture records original test indices and source pixels.

This is a binary two-class classifier, not a ten-digit recognizer. Inputs from
other classes still receive either 3 or 8; no rejection class is trained.
Thresholding and pooling discard information. Decoder agreement proves exact
execution of this small network, not perfect recognition. The instruction
stream and many temporary pictures are expensive compared with ordinary neural
inference. Intel hardware and two software decoders are tested; no universal
hardware compatibility or formal conformance certification is claimed.
