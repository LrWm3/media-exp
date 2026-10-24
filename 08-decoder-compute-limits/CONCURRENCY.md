# Useful inference alongside GPU work

**NVDEC denoising overlaps with CUDA work while retaining about 98–100% of its
baseline throughput. It provides no advantage over CUDA denoising in this
experiment.** The exact CUDA implementation completes the eight-image batch in
about 1.5 ms alone; the current NVDEC runtime takes about 550 ms. Concurrent
completion times favor CUDA for compute pressure and are essentially tied
within variation for memory pressure.

This uses the unchanged trained denoiser from experiment 03, the packed streams
from experiment 07, and eight identical 32×32 inputs for both implementations.
CUDA performs the full eight-channel signed 3×3 convolution, thresholds, and
1×1 mix in one fused kernel. Every output bit matches the frozen independent
model oracle in every run. The decoder computes the same network from its
input picture and prediction instructions.

## Results

Three measured repetitions follow warmups for each of five conditions. Each
combined run completes the **same fixed GPU work plus one inference batch**.
The table reports median time until both jobs finish and the GPU work rate
relative to GPU-only. Rate differences close to 100% are measurement variation.

| GPU pressure | GPU-only completion | GPU + NVDEC completion | GPU + CUDA completion | GPU rate with NVDEC | GPU rate with CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Integer compute | 853.64 ms | 871.01 ms | 854.02 ms | 98.21% | 99.92% |
| Memory traffic | 1,903.51 ms | 1,898.59 ms | 1,901.41 ms | 100.19% | 100.05% |

| Run group | NVDEC inference alone | CUDA inference alone | NVDEC inference under load | CUDA inference under load |
| --- | ---: | ---: | ---: | ---: |
| Integer compute | 554.41 ms | 1.471 ms | 546.23 ms | 0.700 ms |
| Memory traffic | 551.21 ms | 1.534 ms | 547.50 ms | 2.975 ms |

The isolated inference gap is roughly **360–377×** in this runtime comparison.
It is not a ratio of hardware kernel throughput: CUDA uses a persistent service,
whereas NVDEC includes a helper process and decoder session per batch. CUDA
also fuses the whole network into one kernel while decoding executes 136
pictures. Both implementations return the same 8,192 useful output bytes.

NVDEC overlap reduces completion time relative to the estimate obtained by
adding isolated GPU and NVDEC medians: about 1.62× for compute pressure and
1.29× for memory pressure. This estimated serial comparison is not an actual
serial-run measurement, and it does not establish a win over CUDA inference.
The faster CUDA-under-compute-load sample does not imply a universal speedup;
device wake state, scheduling, and short-runtime variation affect these numbers.

## Work and timing boundaries

- **Compute pressure:** 2,700 completed kernels, each covering 262,144 elements
  with 128 iterations of four independent unsigned-integer arithmetic chains.
  Observable output and sample checks prevent dead-work elimination.
- **Memory pressure:** 738 completed kernels, each reading and writing two
  64 MiB buffers for 128 MiB total traffic per unit. Samples are checked after
  completion. Kernel design establishes intended pressure; utilization counters
  were not measured.
- **Fixed work:** a short calibration targets two seconds, then freezes the
  kernel count for all conditions. Actual warmed execution was about 0.85 s for
  compute and 1.90 s for memory; the target is not the measured duration.
- **CUDA inference timing:** command IPC, fresh 8 KiB input H2D, fused kernel,
  8 KiB output D2H, output file I/O, and exact oracle comparison. Context setup,
  allocations, model upload, and PTX loading happen before timing. The service
  uses independent nonblocking streams for inference and background work.
- **NVDEC inference timing:** stream assembly, file I/O, helper/decoder startup,
  parsing, decoding, final-picture readback, extraction, and oracle comparison.
  Each batch decodes 136 pictures and copies only eight final pictures:
  279,552 I420 bytes before extracting the useful 8 KiB. Compilation and PCM
  input packing happen before timing.
- **Combined timing:** ends after inference validation and receipt of the GPU
  completion report. Per-kernel synchronization, result validation, and IPC
  are included. Derived overlap statistics are computed after the timer stops.

The runner rotates condition order and reverses alternate repetitions. Raw
records include 30 measured cases, all source/helper/PTX hashes, input/output
hashes, GPU event stages, and per-unit host intervals. All inferred outputs and
sampled stress results passed validation. No power or clock settings changed.

Every combined run has intersecting inference and GPU-work host intervals;
the minimum NVDEC intersection is 539.80 ms for compute and 545.37 ms for
memory. These intervals and completion times demonstrate concurrent submission
and progress, not simultaneous hardware instructions. CUDA serves as a proxy
for shader workloads here; no graphics API, occupancy, energy, or instruction
trace was measured. CUDA event stages can include scheduling delay and must
not be interpreted as pure active hardware time.

[Readable results](assets/concurrency-summary.json) preserve every run and
summary; the [compressed raw measurements](assets/concurrency-traces.json.gz)
also include every per-unit interval. The summary records both archive and
uncompressed-report hashes. This
workload motivates improving startup/input-output costs before expecting a
useful offload advantage. Larger decoder-resident workloads and sustained
graphics workloads remain unmeasured.

## Reproduce

Prepare experiment 07's bundle and build the existing NVDEC helper first.
The CUDA helper uses the driver API via `libcuda.so.1`, without a toolkit at
runtime. Its build requires a C++17 compiler and a clang build with NVPTX support:

```sh
python3 07-packed-denoising-decoder/prepare.py
sh 08-decoder-compute-limits/build_gpu_workload.sh
python3 08-decoder-compute-limits/concurrency.py \
  --bundle 07-packed-denoising-decoder/out/bundle \
  --weights 03-denoising-decoder/model/weights.json \
  --cuda-helper 08-decoder-compute-limits/.build/gpu_workload \
  --ptx 08-decoder-compute-limits/.build/gpu_workload.ptx \
  --helper decoder_core/.build/nvdec_decode \
  --output 08-decoder-compute-limits/out/concurrency \
  --repeats 3 --seconds 2
```

Alternatively, run the build script with `--ptx-only` on a machine with clang,
copy the PTX and source to the CUDA environment, and build there with
`--host-only`. `CLANGXX`, `CXX`, and `GPU_BUILD_DIR` override build tools/output.
Add `--smoke-only` to check exact CUDA/NVDEC inference and one stress unit before
running measurements. PTX targets the portable instruction subset selected in
the build script; a compatible CUDA driver/device is required.
