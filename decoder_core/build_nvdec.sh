#!/usr/bin/env bash
set -euo pipefail

decoder_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
sdk_root=${NVDEC_BUFFER_API:?Set NVDEC_BUFFER_API to the matching buffer API SDK root}
nvdec_lib_dir=${NVDEC_LIBRARY_DIR:?Set NVDEC_LIBRARY_DIR to the directory containing libnvbufsurface and libnvbufsurftransform}
cuda_root=${CUDA_HOME:-/usr/local/cuda}
decoder_output=${1:-"$decoder_dir/.build/nvdec_decode"}
if [[ $# -gt 1 ]]; then
    echo "Usage: build_nvdec.sh [OUTPUT_BINARY]" >&2
    exit 2
fi
if [[ ! -f "$sdk_root/include/nvbufsurface.h" ]]; then
    echo "Set NVDEC_BUFFER_API to an extracted matching buffer API SDK root." >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$decoder_output")"
read -r -a gst_flags <<< "$(pkg-config --cflags --libs gstreamer-app-1.0 gstreamer-video-1.0)"
"${CXX:-g++}" -std=c++14 -O2 -Wall -Wextra \
    -I"$sdk_root/include" -I"$cuda_root/include" \
    "$decoder_dir/nvdec_decode.cpp" -o "$decoder_output" \
    "${gst_flags[@]}" -L"$nvdec_lib_dir" \
    -Wl,-rpath,"$nvdec_lib_dir" -lnvbufsurface -lnvbufsurftransform
printf '%s\n' "$decoder_output"
