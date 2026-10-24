#!/bin/sh
# Split builds support a CUDA host without toolkit packages: emit PTX wherever
# clang has an NVPTX backend, then compile the small host on the target machine.
set -eu
here=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
out=${GPU_BUILD_DIR:-"$here/.build"}
mkdir -p "$out"
mode=${1:-all}
case "$mode" in all|--ptx-only|--host-only) ;; *) echo 'usage: build_gpu_workload.sh [--ptx-only|--host-only]' >&2; exit 2 ;; esac
if [ "$mode" != --host-only ]; then
    "${CLANGXX:-clang++}" -x cuda -std=c++17 -O3 --cuda-device-only \
        -nocudainc -nocudalib --cuda-gpu-arch=sm_53 -DGPU_KERNELS -S \
        "$here/gpu_workload.cu" -o "$out/gpu_workload.ptx"
fi
if [ "$mode" != --ptx-only ]; then
    "${CXX:-c++}" -x c++ -std=c++17 -O2 -Wall -Wextra -pthread \
        "$here/gpu_workload.cu" -ldl -o "$out/gpu_workload"
fi
