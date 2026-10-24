# 11 — Retained reference and completion experiment

Experiment 10 found no practical elapsed-time win, even after omitting host
readback. Every request still decoded fresh inputs and reset references with
an IDR fence. This experiment separates retention from completion behavior.

## Predeclared hypotheses

1. Loading one input set once and generating many independently requested
   outputs amortizes input decode and completion overhead. Compile distinct
   atlas fields and supplied-motion interpolation fields through the shared IR.
   Every named output must match native CPU and complete I420 decoder output.
2. The existing runtime's IDR-per-request contract hides a usable continuous
   path. Add a separate bounded append/collect protocol with one initial IDR,
   no intermediate reset, and one final EOS. Append acknowledgment is never
   decode completion. Bounded collection reports partial progress explicitly.
3. Declaring sixteen reference slots when the actual schedule needs only one,
   two or three may increase output holding. Add opt-in schedule-derived SPS
   reference/DPB bounds, matching memory-management commands. Preserve default
   bitstreams, source lifetimes and exact pixels. Compare both declarations.

## Fixed corpus and measurements

Use active sizes 128 and 512, 1/8/32 distinct named outputs, atlas and two-input
interpolation, and both original/tight DPB declarations. Four changing fixtures
are scene, noise, zero and 255. Inputs load once per finite sequence. Every
prediction reads the original inputs; no output pixels are encoded as answers.
Use whole-picture prediction slices and the existing native/unrolled CPU
kernels. Compare each field with its exact semantic graph and native CPU spec.

First probe output availability at paused AU boundaries using bounded polling.
Record submitted/decoded/pending counts and exact frames, including final EOS
drain. Additional prediction lookahead counts as extra work, not immediate
completion. A timeout is a recorded availability result, not a silent success.
Do not disable DPB management for arbitrary multi-reference P/B programs.

Then measure complete finite output sets against CPU with input decode and
all requested host results included. Retain a separate completion-only
measurement as an unequal-interface diagnostic. Five rotated repeats and four
changing inputs are the default. Report elapsed and process CPU separately,
including allocation/setup/close boundaries and every fence or EOS explicitly.
If an immediate retained-state path works, measure it separately with exact
boundary counts; otherwise document the minimum observed pending work and
retain the existing fenced path for fair complete-output comparisons.

## Limits and decision rules

This is decoded-reference retention, not arbitrary CUDA-surface import.
No consumer-surface handoff or energy measurement is implied by completion.
No SDK installation, power/clock/priority changes, or device identity in public
records. Source, artifact, helper and CPU build hashes bind measurements.

A throughput gain with a long precompiled batch does not prove interactive
latency, a useful consumer pipeline or universal CPU optimality. If favorable
results depend on omitted output work, label that dependence and investigate a
real consumer before claiming an application win. If software contracts still
prevent immediate completion, identify that boundary without claiming a small
hardware change is sufficient or estimating silicon cost.
