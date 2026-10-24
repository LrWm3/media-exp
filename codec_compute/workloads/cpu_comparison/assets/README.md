# Representative CPU/decoder fixtures

This directory retains fixture 0 for all twelve size/task combinations. The
270 referenced payloads total 28,545,841 bytes and include input planes, input
access units, prediction units, IDR fences, complete streams, ordinary streams
without fences, decoded I420 and compact output references.

`bundle.json` describes the retained 12 streams / 194 pictures and their 36
checks from the original local export. `full-bundle.json` is the byte-identical
original index for 96 streams / 1,552 pictures / 288 local checks; payloads for
its other seven fixtures per case are omitted here. Selection provenance keeps
the original counts, fixture indices and complete-index hash explicit. Hardware
benchmark reports refer to that full index, rather than the trimmed bundle.

The original export records matching full decoded I420 from FFmpeg, OpenH264
and VA-API, and independently computed CPU outputs. Curation rechecks every
retained artifact hash, framing, named output, guard and chroma plane. It also
binds CPU kinds, round counts, motion fields and recurrence dependencies to the
saved semantic graphs. Curation performs no additional decoder run.

The [128-pixel sheet](reference-outputs-128.png) and
[256-pixel sheet](reference-outputs-256.png) show actual output crops from saved
decoded I420. Every pixel is displayed at 1:1 scale, with no interpolation or
contrast adjustment. Black pixels have value zero. Individual unscaled PNGs
are under `panels/`. These pictures are local reference evidence; consult the
separate benchmark report for hardware validation and timing.

`capture.json` binds the rendering script, both indexes, full I420 sources,
compact output references, individual panels and contact sheets by SHA-256.
Reproduce from the full original export:

```sh
python -m codec_compute.workloads.cpu_comparison.capture \
  codec_compute/artifacts/cpu-comparison \
  --output codec_compute/workloads/cpu_comparison/assets --fixture 0
```

The original index SHA-256 is
`f7758a7f0b7921ee1bf6b447f32701bbde11696739b73ae9b70e871d402d3d1b`.
Its source revision remains `ed20a7185d360a0a0560a1733a254c731d55cae0`.
