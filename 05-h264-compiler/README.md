# A tiny Python → H.264 compiler

Write a restricted Python image kernel, compile it once, and let an ordinary
video decoder execute its arithmetic. One shared graph, instruction selector,
reference scheduler, and backend now serve variants of all four earlier experiments.

```python
import h264

@h264.kernel
def neighbor_or(image):
    right = h264.shift(image, dx=1, dy=0)
    below = h264.shift(image, dx=0, dy=1)
    return h264.threshold(h264.add(right, below), 1)

program = h264.compile(neighbor_or, width=32, height=32)
image = [[0] * 32 for _ in range(32)]
image[16][16] = 1
program.save("compute.h264", image)  # also writes compute.json
result = program.verify(image, backend="ffmpeg")["output"]
```

Run from this directory, or add it to `PYTHONPATH`. Compilation receives symbolic
images and static dimensions; input pixels enter only during encoding.
The prediction bytes in `program.tail` are reusable with different inputs.
`verify()` decodes and separately checks every named output pixel against an
integer graph interpreter. `decode()` returns actual decoder output without
that reference calculation; `read(raw_yuv)` inspects an existing decode.

## Native image operations

Version 0.2 adds independent input pictures, explicit saturating affine
arithmetic, per-block gathers, and exact quarter-pixel interpolation:

```python
program = h264.compile(
    lambda a, b: h264.saturating_add(a, b),
    width=64, height=64,
    input_ranges={"a": (0, 255), "b": (0, 255)},
)
program.save("add.h264", {"a": image_a, "b": image_b})
```

This produces **three pictures**: two literal input pictures and one B-picture
computing `min(255, a+b)`. Loading the second input preserves the first reference.

[Six native-operation demos](native_ops/README.md) include addition,
brightness/contrast, tile permutation, blockwise warp, fractional translation,
and a chain retaining both original inputs. Each includes an executable stream,
manifest, and captures from actual decoding. The [benchmark](native_ops/BENCHMARK.md)
separates host input packing from decoding and compares fresh inputs with
resident references. Tests run on FFmpeg, OpenH264, and Intel VA-API.
[NVDEC validation](native_ops/nvdec/README.md) also passed all 34
hardware cases: 6,116 pictures matched byte-for-byte, including all four compiled
experiments. Desktop CUVID and SDK 13.1 direct-array output remain untested.

![Saturating addition captured from Intel decoding](native_ops/assets/saturating-add.png)

[Experiment 06](../06-motion-mask-decoder/README.md) combines frame differences,
thresholding, and morphology into motion masks. It also adds one-session batch
execution and selective NVDEC output readback to the shared runtime.

## Original binary example

![Shifted OR captured from Intel hardware decoding](assets/shifted-or.png)

This example needs **four pictures**: visible seed, binary normalization, fused
shifted addition, and threshold. Results are logical 0/1 values; the capture
enlarges them and maps 1 to white. The executable stream contains computation
pictures, not a precomputed presentation animation.

## Compiled experiments

| Variant | Result | Original → compiled pictures |
| --- | --- | ---: |
| [01 Life](../01-conways-decoder-of-life/compiled/README.md) | Every generation matches; 40-generation capture | 561 → 922 |
| [02 Maze](../02-maze-flood-fill-decoder-solver/compiled/README.md) | Same flood states, distances, and 50-hop route; 52 steps | 261 → 560 |
| [03 Denoiser](../03-denoising-decoder/compiled/README.md) | All 256 held-out feature/output maps match; 3.2932% pixel error | 98 → **74** per image |
| [04 MNIST](../04-binary-mnist-classification-decoder/compiled/README.md) | All 1,984 prediction/hidden records match; **94.96%** accuracy | 1,802 → 1,826 per 8-digit batch |

Each variant includes a kernel, CLI, tests, executable demo, instruction manifest,
and actual hardware capture. Existing implementations and trained weights remain
available unchanged. The denoiser benefits from common-expression elimination;
the compiler is not yet as compact as every specialized schedule.

## Run and inspect

Python 3.10+ and FFmpeg suffice. Optional OpenH264 and Intel VA-API execution
use the same GStreamer backends as earlier experiments. The optional
[NVDEC backend](native_ops/nvdec/README.md#reproduce) uses a small C++ helper
to preserve exact full-range bytes during hardware readback.

```sh
python3 run.py --preview
python3 run.py --decoder vaapi --preview
python3 -m unittest -v
```

`program.h264` contains the seed and prediction instructions. Its sibling
`program.json` records the graph, proven ranges, scheduled pictures, physical
references, refresh copies, named output frame numbers, and hashes.
See the [small example manifest](assets/shifted-or.json).

See [API and exact semantics](DESIGN.md) and [validation](VALIDATION.md) for
compiler tests, exhaustive byte arithmetic and interpolation checks, and
comparisons with the original neural models.
