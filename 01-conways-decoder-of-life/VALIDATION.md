# Recorded validation

Run date: 2026-10-23.

Environment: Python 3.12, FFmpeg 6.1.1, GStreamer 1.24.2. Hardware backend:
`vah264dec`, reported as “VA-API H.264 Decoder in Intel(R) Gen Graphics”, using
`/dev/dri/renderD128` (PCI vendor `8086`, device `a7a1`).

## Integration suite

```sh
python3 -m unittest -v
```

Result: **9 tests passed, 0 skipped**, in 3.084 seconds.

| Check | Result |
| --- | --- |
| All 512 neighborhoods, FFmpeg | Pass |
| All 512 neighborhoods, Intel hardware | Pass |
| All 512 neighborhoods, OpenH264 | Pass |
| Six patterns × 25 generations, FFmpeg | Pass |
| Random board × 25 generations, Intel hardware | Pass |
| Replacement seed with byte-identical future instructions | Pass |
| Zero-generation seed | Pass |
| 5,000-generation reference-distance check (structural only) | Pass |
| Rejection of three level-limit violations | Pass |

## Demo stream

```sh
python3 run.py --generations 120 --verify --preview --output out/r-pentomino.h264
```

- Board: 64×64 cells, initially an R-pentomino.
- Coded picture: 96×96 pixels, including the permanent zero guard.
- Stream: 1,033,864 bytes, High profile level 3.1.
- Pictures: 1,681, representing the seed plus 120 × 14 instruction pictures.
- FFmpeg: all 121 generation states match the independent Life oracle.
- Intel VA-API: same stream, all 121 generation states match.
- OpenH264: same stream, all 121 generation states match.
- Every decoder preserves zero guard and neutral chroma in every picture.
- Hardware and OpenH264 also produce byte-identical complete raw output,
  including the intermediate calculations: 23,238,144 bytes each.
- GIF preview: 110,847 bytes, generated from FFmpeg-decoded generation pixels.

Stream SHA-256:

```text
3942ce939d9d6ea007aa3d3b73d586d4f6ebc2a866561cc083b3d70104aa4a27
```

Hardware/OpenH264 raw I420 output SHA-256:

```text
f7d40bf735225c9a0235b5f825839dd82d2a3ee112a94dd6ea599238057620a7
```

Optional backend runs can be reproduced using `run.py --verify --decoder vaapi`
and `run.py --verify --decoder openh264`; both generate the same default program.
The bitstream generator is deterministic and never calls the Life oracle.

## Scope

Successful decoding alone does not certify standards conformance. A separate
review against the [ITU-T H.264 August 2024 edition](https://www.itu.int/rec/dologin_pub.asp?id=T-REC-H.264-202408-S!!PDF-E&lang=e&type=items)
checked weighted-prediction ranges and pair sums, reference-list handling, High
profile PCM support, VUI restrictions, level 3.1 dimensions and buffer capacity,
slice rate, and the first-access-unit size bound. This led to a single PCM seed
slice and explicit input limits. No official conformance suite was run. Other
hardware devices and decoding across 16-bit frame/POC counter wraps remain untested.
