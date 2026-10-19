#!/usr/bin/env bash
# Compare native CUDA and native CPU finals against Formal finals with the
# strict raw comparator.
#
# usage: validate.sh <case-dir> [native CUDA options...]
#
# <case-dir> holds pairs <case>.fadv2d (Formal --dump-input) and
# <case>-formal.fadv2d (Formal --dump-final-raw on the same input).  Every
# case runs the native CPU oracle and native CUDA on the shared input; the
# comparator checks all three finals (Formal, CUDA, CPU oracle) with the
# unchanged tolerances, fixed boundaries and, with EVOLUTION=1 (default),
# meaningful evolution.  Set SKIP_CPU_ABOVE=<cells> to skip the slow CPU
# oracle above that many interior cells (the comparison is then Formal
# versus CUDA only).  PYTHON selects the interpreter (default python3).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
formal_root="$(cd "$here/../.." && pwd)"
case_dir="$1"
shift
binary="$here/build/formal_advection"
python="${PYTHON:-python3}"
skip_cpu_above="${SKIP_CPU_ABOVE:-300000}"
evolution=()
if [[ "${EVOLUTION:-1}" == 1 ]]; then
    evolution=(--evolution)
fi
lock=()
if [[ -n "${GPU_LOCK:-}" ]]; then
    lock=(flock "$GPU_LOCK")
fi

status=0
for input in "$case_dir"/*.fadv2d; do
    case "$input" in
        *-formal.fadv2d) continue ;;
    esac
    name="$(basename "$input" .fadv2d)"
    formal="$case_dir/$name-formal.fadv2d"
    [[ -f "$formal" ]] || continue
    read -r nx ny steps _ < <(sed -n 2p "$input")
    out="$case_dir/out"
    mkdir -p "$out"
    cuda_final="$out/$name-cuda.bin"
    "${lock[@]}" "$binary" --mode cuda --input "$input" --dump-bin "$cuda_final" "$@" \
        > "$out/$name-cuda.log"
    cpu_args=()
    if (( nx * ny <= skip_cpu_above )); then
        cpu_final="$out/$name-cpu.bin"
        if [[ ! -f "$cpu_final" || "$cpu_final" -ot "$binary" ]]; then
            "$binary" --mode cpu --input "$input" --dump-bin "$cpu_final" \
                > "$out/$name-cpu.log"
        fi
        cpu_args=(--cpu-final "$cpu_final")
    fi
    echo "== $name ($nx x $ny, $steps steps) $*"
    if ! "$python" "$formal_root/example/scripts/compare_2d_raw_fields.py" \
        "$input" "$formal" "$cuda_final" --cells "$nx,$ny" --steps "$steps" \
        --rtol 1e-6 --atol 1e-8 "${evolution[@]}" "${cpu_args[@]}"; then
        echo "FAILED: $name"
        status=1
    fi
done
exit "$status"
