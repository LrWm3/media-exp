#!/usr/bin/env bash
set -euo pipefail

decoder_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
pcm_output=${1:-"$decoder_dir/.build/libpcm_native.so"}
if [[ $# -gt 1 ]]; then
    echo "Usage: build_pcm.sh [OUTPUT_LIBRARY]" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$pcm_output")"
"${CXX:-c++}" -std=c++17 -O3 -Wall -Wextra -Wpedantic -fPIC -shared \
    "$decoder_dir/pcm_native.cpp" -o "$pcm_output"
printf '%s\n' "$pcm_output"
