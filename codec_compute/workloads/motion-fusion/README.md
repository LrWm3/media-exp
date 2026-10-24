# Folding spatial reads into H.264 prediction

The shared compiler now uses a consumer's motion vector to read a shifted image
directly. It removes four temporary decoded pictures from each spatial workload,
while preserving every named output and all rounding and clipping steps.

At 32×32 logical pixels (64×64 coded with zero guards):

| Workload | Compute pictures, before → after | Prediction bytes, before → after | Peak live values, before → after |
| --- | ---: | ---: | ---: |
| Binary cross dilation | 8 → 4 | 1,262 → 692 | 3 → 2 |
| Nested rounded smoothing | 8 → 4 | 1,262 → 688 | 3 → 2 |
| Cross minimum, maximum, and range | 21 → 17 | 3,553 → 2,979 | 7 → 4 |

Every job also uploads one input picture. These are static savings, not measured
latency improvements. The existing workload capture illustrates the unchanged
task mathematics; this comparison records both plans and both input fixtures.

Only unnamed zero-border shifts consumed exclusively by supported scalar
operations are eligible. Shifts feeding another shift stay materialized:
shifting out of an image and back must not restore cropped pixels. Named
intermediates stay observable. Multi-stage arithmetic moves only its initial
reads; intermediate clipping and threshold stages remain stationary.

The compiler schedules both plans and compares actual picture counts,
prediction bytes, and distinct named-output pictures. It keeps the baseline if
any cost increases or the candidate cannot be scheduled. For example, two
named copies of one shared shift should still alias one picture; duplicating
the moving read would cost two pictures and is rejected. This is a conservative
whole-plan choice, not a search over every possible subset of fused shifts.

```python
optimized = media.compile_graph(graph, target="h264")
baseline = media.compile_graph(graph, target="h264", h264_fuse_shifts=False)
print(optimized.manifest()["optimizations"]["shift_read_fusion"])
```

Reproduce from the repository root:

```sh
python3 -m codec_compute.workloads.fusion
python3 05-h264-compiler/optimizations/ir/validate.py \
  codec_compute/artifacts/motion-fusion --backend nvdec \
  --helper decoder_core/.build/nvdec_decode
```

The first command creates both plans, verifies the exact saved streams with
available local H.264 decoders, and compares named outputs against independent
task formulas. It also creates the portable suite for the second command;
copy the prepared bundle and decoder validator to an NVDEC environment first.
Full I420 references, including guards and chroma, remain in the regenerated
bundle under ignored `codec_compute/artifacts/motion-fusion`.

The recorded [comparison](assets/comparison.json) passes **12 saved streams and
36 local decoder checks**, using FFmpeg, OpenH264, and VA-API. Both plans and
both fixtures produce identical named output pixels. The
[suite](assets/suite.json) indexes all **136 pictures** and exact output hashes.
Saved streams, manifests, and detailed per-plan reports are included under
`assets`; regenerate the bundle for full I420 references and named output planes.
Source hashes and commit identify the compilation snapshot.

[NVDEC validation](assets/nvdec-validation.json) also passes all **12 streams**:
**136 pictures and 835,584 full I420 bytes**, including intermediate pictures,
zero guards, neutral chroma, and every named output. This validates the actual
fused streams on NVDEC; it is separate from the temporal timing experiment.
