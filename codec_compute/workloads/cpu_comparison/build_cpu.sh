#!/usr/bin/env bash
set -euo pipefail
cpu_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cpu_output=${1:-"$cpu_dir/.build/libcpu.so"}
if [[ $# -gt 1 ]]; then
    echo 'Usage: build_cpu.sh [OUTPUT_LIBRARY]' >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$cpu_output")"
"${CXX:-c++}" -std=c++17 -O3 -DNDEBUG -fPIC -shared -Wall -Wextra \
    "$cpu_dir/cpu.cpp" -o "$cpu_output"
printf '%s\n' "$cpu_output"
