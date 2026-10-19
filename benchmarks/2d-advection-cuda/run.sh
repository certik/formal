#!/usr/bin/env bash
# Build the native C + CUDA reference and run it.
#
#   bash benchmarks/2d-advection-cuda/run.sh [formal_advection options...]
#
# NVCC selects nvcc (default: nvcc on PATH, else /usr/local/cuda/bin/nvcc),
# CUDA_HOST_CXX the host C++ compiler passed to nvcc -ccbin (default: nvcc's
# choice) and CUDA_ARCH the target architecture (default: native).  CUDA mode
# is opt-in (FA_ALLOW_CUDA=1).  When GPU_LOCK names a lock file, every CUDA
# run is serialized through `flock "$GPU_LOCK"` so timings never overlap with
# other users of the GPU.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -eq 0 ]]; then
    set -- --mode cpu --cells 11,11 --steps 2 --dt 1e-4
fi

mode=cpu
mode_seen=0
expect_mode=0
for arg in "$@"; do
    if ((expect_mode)); then
        mode="$arg"
        expect_mode=0
    elif [[ "$arg" == "--mode" ]]; then
        if ((mode_seen)); then
            echo "--mode may be specified only once" >&2
            exit 2
        fi
        expect_mode=1
        mode_seen=1
    fi
done
if ((expect_mode)); then
    echo "--mode must be followed by cpu or cuda" >&2
    exit 2
fi
if [[ "$mode" != cpu && "$mode" != cuda ]]; then
    echo "--mode must be cpu or cuda" >&2
    exit 2
fi
if [[ "$mode" == cuda && "${FA_ALLOW_CUDA:-0}" != 1 ]]; then
    echo "CUDA mode is guarded; set FA_ALLOW_CUDA=1 to proceed." >&2
    exit 1
fi

nvcc="${NVCC:-}"
if [[ -z "$nvcc" ]]; then
    nvcc="$(command -v nvcc || true)"
    if [[ -z "$nvcc" && -x /usr/local/cuda/bin/nvcc ]]; then
        nvcc=/usr/local/cuda/bin/nvcc
    fi
fi
if [[ -z "$nvcc" || ! -x "$nvcc" ]]; then
    echo "nvcc not found; install the CUDA toolkit or set NVCC." >&2
    exit 2
fi
host=()
if [[ -n "${CUDA_HOST_CXX:-}" ]]; then
    host=(-ccbin "$CUDA_HOST_CXX")
fi

build_dir="$here/build"
binary="$build_dir/formal_advection"
mkdir -p "$build_dir"
"$nvcc" -O3 -std=c++17 -arch="${CUDA_ARCH:-native}" "${host[@]}" \
    -Xcompiler -Wall,-Wextra -lineinfo \
    "$here/formal_advection.cu" -o "$binary"

if [[ "$mode" == cuda && -n "${GPU_LOCK:-}" ]]; then
    exec flock "$GPU_LOCK" "$binary" "$@"
fi
exec "$binary" "$@"
