# Native-operation benchmark

64 x 64 active pixels; 32 saturating additions; 3 timed repetitions per decoder. All times are wall-clock medians.

Hardware: 00:02.0 VGA compatible controller [0300]: Intel Corporation Raptor Lake-P [Iris Xe Graphics] [8086:a7a1] (rev 04).

The resident workload loads two inputs once and keeps the accumulator and increment image inside the decoded-picture reference buffer. Fresh pairs reload both operands for each addition. They are measured with either one decoder start per pair or one decoder session for all pairs. All three compute the same final pixels. Fresh operand fixtures are precomputed on the host; this fixed-workload benchmark does not model a host feedback loop.

## Host construction

| Phase | Pair kernel (ms) | Resident chain (ms) |
| --- | ---: | ---: |
| Compile graph, lower, schedule | 0.039423 | 0.544269 |
| Build prediction bytes once | 1.039903 | 36.658563 |
| Read cached bytes | 0.000059 | 0.000063 |
| Encode/validate two PCM inputs | 1.312766 | 1.303262 |
| Encode inputs + concatenate cached tail | 1.338027 | 1.395506 |

## Input loading

| Build complete input-bearing stream | Median (ms) |
| --- | ---: |
| Encode all 32 fresh pairs + concatenate cached tails | 44.705 |
| Encode one resident pair + concatenate cached tail | 1.390 |

Operand fixtures already exist before these timers start. Both measurements include input validation, PCM encoding, and stream concatenation. Prediction tails are cached.

## Complete workloads

| Quantity | Fresh pairs, separate sessions | Fresh pairs, one session | Resident chain |
| --- | ---: | ---: | ---: |
| Logical additions | 32 | 32 | 32 |
| PCM input pictures | 64 | 64 | 2 |
| Prediction pictures | 32 | 32 | 34 |
| Refresh copies | 0 | 0 | 2 |
| All decoded pictures | 96 | 96 | 36 |
| Decoder process starts | 32 | 1 | 1 |
| Encoded input bytes | 1,055,176 | 1,055,176 | 32,982 |
| Prediction bytes | 20,000 | 20,000 | 21,514 |
| All stream bytes | 1,075,176 | 1,075,176 | 54,496 |
| All YUV readback bytes | 1,327,104 | 1,327,104 | 497,664 |

## Decode, readback, and output validation

| Decoder | Fresh / separate sessions (ms) | Fresh / one session (ms) | Resident (ms) | Exact result |
| --- | ---: | ---: | ---: | --- |
| ffmpeg | 1195.298 | 48.667 | 44.890 | pass |
| vaapi | 1190.479 | 64.463 | 63.596 | pass |
| openh264 | 683.950 | 34.281 | 26.581 | pass |

Each decoder measurement includes external process startup, parsing, decoding, all-picture YUV readback, and Python output/guard validation. Stream construction and prewriting benchmark files are excluded from these decode measurements and measured separately above. GStreamer backends also include writing and reading a temporary YUV file. This is an application pipeline measurement, not an isolated device-kernel benchmark.

The separate-session comparison includes fewer decoder startups. The single-session comparison holds startup count at one while comparing repeated input loads/IDR resets against resident references. Both comparisons include fewer PCM pictures and less readback in the resident case. They do not establish GPU arithmetic throughput or superiority over CPU/vectorized addition. No NVIDIA hardware or NVDEC was tested.

Prediction bytes are input-independent and reused without rebuilding. Reference refresh copies extend live values beyond the 16-picture DPB window; they consume both pictures and stream bytes. The workload report counts these copies explicitly. The runtime still loads inputs using ordinary H.264 PCM pictures; it cannot directly inject host buffers into decoder references.

See `benchmark.json` for every timing sample, stream size, output hash, and machine provenance.
