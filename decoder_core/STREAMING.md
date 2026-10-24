# Bounded retained-reference streaming

`NVDECStream` is an experimental low-level interface for one compiler-produced
H.264 sequence. It keeps the decoder pipeline and reference pictures alive
across calls. The first AU contains SPS/PPS/IDR; every later AU is a non-IDR
picture from the same precompiled global schedule. Only final close sends EOS.

```python
from decoder_core.stream import NVDECStream

# Compile the entire bounded graph first. Input frame numbers, picture order
# and long-term slot lifetimes must remain continuous across all appends.
units = program.native.access_units(inputs)
writer = program.native.writer
received = []
with NVDECStream(writer.width, writer.height, helper="/path/to/nvdec_decode") as stream:
    for unit in units:
        stream.append((unit,))
        first = stream.stats["decoded_frames"]
        target = stream.stats["submitted_frames"]
        received.extend(stream.collect(target, range(first, target), wait_ms=50))
    first = stream.stats["decoded_frames"]
    received.extend(stream.close(range(first, len(units))))

# Each item is (absolute_picture_index, complete_I420_bytes).
assert [index for index, _ in received] == list(range(len(units)))
```

An append acknowledgment only confirms acceptance into the input pipeline.
It cannot certify completion. `collect(target, selected_frames, wait_ms=...)`
waits for samples through a cumulative picture count and can return partial
progress when its bounded interval expires. Re-submit only the still-pending
selected indices on the next collect. Samples omitted from a selection are
discarded and cannot be retrieved later. An empty selection measures output
availability without a host pixel copy; it does not expose a GPU consumer.

The `decoded_frames` counter means output samples actually pulled from
appsink. `pending_frames` is submitted minus observed output. Neither counter
reveals the internal hardware execution state of a picture that the driver or
plugin has not released. This distinction matters when diagnosing buffering.

`close(selected_frames=...)` sends the sole EOS, drains every pending real
picture, optionally copies selected pending pictures, validates final counters
and delayed error logs, and destroys the pipeline. Final EOS therefore cannot
serve as an interactive completion operation while preserving references.

The separate NVST v1 protocol does not change resident NVDS v2 `NVDECSession`
requests or file decoding. Build the current helper explicitly. Every session
is fixed-shape, sequential, capped at 32,767 total pictures and 128 MiB
outstanding compressed input, with at most 128 MiB returned per request.
Per-collect wait is 0..60,000 ms; transport and close use the caller's finite
timeout. A successful partial collect and input validation failure before
transmission keep the stream usable; transport, decoder and protocol failures
poison it. One `wait_ms=0` collect polls at most one available sample.

Images remain owned by the decoder. This API neither imports arbitrary
CUDA/input surfaces nor transfers output surface ownership. Appending an
unplanned prediction after the compiled schedule ends is invalid: the final
output may replace a source's slot. Compile all intended dependencies together.

The paired [experiment 11](../11-retained-reference-streaming/README.md) checks
every decoded I420 sample and records paused-boundary availability separately
from complete finite-sequence timings. `h264_tight_dpb=True` narrows normative
reference declarations; it does not disable DPB management. No low-latency
decoder switch was enabled for multi-reference weighted B predictions.
