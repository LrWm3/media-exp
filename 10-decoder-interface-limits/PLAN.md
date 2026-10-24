# Plan: find a niche or identify the missing interface

Start from the measured result in experiment 09: compatible row-header merging
saves process CPU on longer chains, but even the 64-stage case remains slower
than native CPU. That does not yet establish a useful application.

## Hypotheses and interventions

1. **Prediction-slice overhead hides useful arithmetic throughput.** Extend the
   shared H.264 target with opt-in bands of prediction rows, including one full
   picture. Implement the actual cross-row motion-vector predictor, preserve
   default bytes and fall back when headers differ. Compare identical pixels
   with 1-row, 4-row and whole-picture slices.
2. **A small practical primitive becomes competitive.** Retest patch routing
   and supplied-motion two-frame interpolation, plus a copy control. Keep
   8/32/64-stage recurrences as throughput probes, explicitly synthetic.
3. **The interface, rather than the arithmetic, blocks a niche.** Separately
   measure fresh host inputs/output, pre-encoded inputs/output, and pre-encoded
   inputs with decode completion but no readback. The last two modes omit work
   and must not be presented as equivalent application wins. All still decode
   their input pictures; none demonstrates arbitrary reference-surface import.
4. **Experiment 09 should be the entry point.** Number the existing shared IR
   without moving its import package or pinned historical artifacts. Deprecate
   the experiment 05 frontend for new kernels, document migration gaps and
   preserve its still-used backend/packing components.

## Fixed comparison

Use 512×512 active images, four changing scene/noise/zero/255 fixtures, and five
rotated timing repeats. Native CPU kernels use optimized native/unrolled builds;
chains retain the stronger separable-filter implementation from the earlier
CPU comparison. Reusable allocation/setup is excluded for both implementations.
Fresh-host decoder timings include native PCM serialization, decoding, counted
completion fence, selected I420 output transfer and exact comparison. CPU
returns compact luma bytes and checks them. Report process CPU and elapsed time
separately, retain all raw samples, and avoid attributing differences between
independent mode medians to isolated hardware stages.

Every layout must match complete decoded I420 across available local decoders
and live NVDEC. Completion-only timing needs full-pixel preflight and per-trial
postflight on the same fixtures, outside timing. Its zero copied-frame count,
all decoded-picture counts, final fence drain and helper capability must be
verified. It discards surfaces; it is not a device-consumer implementation.

No power/clock/priority changes or SDK installation. Public records identify the
hardware backend as NVDEC. Pin sources, inputs, native build flags/binaries and
raw reports. Capture actual decoded outputs and keep representative payloads.

## Decision rules

A CPU comparison qualifies only with the same complete requested result. A
separated five-trial range is stronger evidence than an isolated median, though
neither proves a universal advantage. Keep performance wins workload-specific.
A synthetic deep-chain win does not establish useful image-processing demand.

If eliminating host serialization and readback still loses to CPU, an output
interop change alone has not unlocked that primitive on this path. If it wins,
measure a real device consumer before claiming an application benefit. Imported
reference surfaces and immediate completion without a reset/fence remain
separate, untested capabilities; the diagnostic does not simulate them.

Rank possible interface changes using these observations: batched prediction
commands, decoder-reference surface import, direct consumer surfaces, and
completion signaling. Distinguish API/driver possibilities from silicon work.
Without implementation/RTL data, do not call any proposed hardware change small
in area, power or engineering effort. Accept a null result for current practical
primitives and state exactly which missing capability could change it.
