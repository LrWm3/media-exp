# Which interface or hardware change would matter?

Separate three questions: whether a primitive is mathematically expressible,
whether today's software feeds it efficiently, and whether an application saves
useful resources. A slice-count improvement answers part of the second question.
It does not measure arithmetic-unit utilization or the area of a proposed chip.

## Priority after measurement

The [paired results](README.md#measurements) put software instruction delivery
first: whole-picture slices cut 64-stage latency from 216.27 to 96.82 ms,
without a hardware change. Four-row bands already achieve most of that gain.
The next question is input/reference ownership and completion: even prepared
inputs with no output readback still take 9.38 ms for atlas routing and
14.02 ms for interpolation, versus CPU 0.17 and 2.14 ms. Those diagnostics
still decode inputs and submit fences, so they do not isolate either cost.

Direct output interop alone has no demonstrated standalone win here. Combine
it with a genuine resident-input experiment and a real consumer before asking
for new silicon. A smaller atlas output remains a useful geometry hypothesis,
but our full-canvas routing test does not measure it. Longer resident chains
already save process CPU; application relevance and energy remain open.

## Candidates and tests

| Candidate | Concrete change | What this experiment can establish | What remains untested |
| --- | --- | --- | --- |
| Batched prediction submission | Submit a picture's compatible block predictions together, with one completion | Identical row/4-row/full-picture streams isolate layout sensitivity | Which parser, firmware, driver or hardware stage causes the difference |
| Direct consumer surface | Let a GPU consumer read the decoded output in place | Completion-only runs omit all result readback and host comparison | Surface lifetime, synchronization, consumer execution and useful end-to-end speed |
| External reference import | Bind a compatible existing image surface as a prediction reference | Prepared-input runs remove PCM generation, but still decode every input | Arbitrary input import, reference metadata initialization and actual eliminated decode cost |
| Immediate completion with retained references | Signal the final compute picture without an extra IDR and preserve DPB state | Current reports count the real completion fence explicitly | A fence-free runtime and persistent state across independent requests |
| Independent output rectangle | Gather an ROI atlas into a smaller destination surface | Current atlas checks exact routing within the original coded canvas | Different input/output geometries and reduced output allocation/bandwidth |

These are interface proposals. Some may be achievable in software or firmware;
others may require hardware changes. No RTL, gate-count, power or validation
estimate exists here, so none is claimed to be a small silicon modification.

## Existing interfaces worth checking first

NVIDIA's SDK 13.1 documents application-allocated opaque CUDA arrays registered
with `cuvidRegisterDecodeSurfaces`, asynchronous decode, and direct consumer
access without the traditional map/unmap output path. Thus direct output is
already a documented API direction, not automatically a request for new silicon.
That documentation does not establish that an arbitrary pre-filled image can
become a valid prediction reference without a decode. See the official
[NVDEC programming guide, section 5](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-video-decoder-api-prog-guide/index.html#reference-traditional-vs-opaque-data-flow).

The Linux stateless decoder interface exposes per-request codec state and lets
clients identify decoded capture buffers as references. It also specifies
reference-buffer lifetime requirements. This is a useful existing model for
explicit scheduling and surface ownership; it does not promise arbitrary
CUDA-tensor injection on our runtime. See the kernel's
[stateless decoder buffer-management contract](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-stateless-decoder.html#buffer-management-while-decoding)
and [H.264 DPB controls](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/ext-ctrls-codec-stateless.html).

The reproducible [availability probe](probe_runtime.py) inspects library symbols
and plugin properties only. Missing symbols constrain the installed software
path; they do not prove hardware impossibility. Present symbols would still
require a device-capability query and a working consumer test. No settings or
SDK packages are changed by the probe.

The [recorded probe](assets/runtime-probe.json) found the existing stateful
NVDEC plugin, but no loadable NVDECODE library or direct-array symbols on the
tested software installation. We therefore did not execute the direct CUDA
array API or a GPU consumer. Changing the software stack could change this
availability result without changing the underlying silicon.

## A falsifiable next interface experiment

A useful prototype would accept two compatible surface handles, a destination
handle, an externally supplied 16×16 motion field for each source, and exact
weights/offsets/denominator. It would run one fused resample/blend and signal
completion while preserving requested references. This reuses the demonstrated
prediction arithmetic; it does not add arbitrary matrix multiplication.

Compare that full source-to-consumer operation with tuned CPU and GPU kernels,
including necessary imports, synchronization and ownership transitions. Use
actual patch-atlas or motion-interpolation consumers. If it only wins after
ignoring those costs, reject the niche. If software whole-picture slices already
remove the dominant cost, prioritize software integration before chip changes.
