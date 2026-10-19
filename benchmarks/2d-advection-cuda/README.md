# Native 2D advection-diffusion reference (C + CUDA)

This directory contains the native C + CUDA reference implementation of the
Formal 2D advection-diffusion benchmark for NVIDIA GPUs.  It is the CUDA
counterpart of [`../2d-advection-metal`](../2d-advection-metal/README.md) and
the performance target for LFortran-compiled Formal (`--gpu=cuda`).  Like the
Metal reference it lives under `benchmarks/`, outside FPM's example
discovery; no FPM is involved.

The CPU path is the FP32 numerical oracle, ported unchanged from the Metal
reference.  All paths implement the corrected order-4 mimetic gradient,
center-to-face interpolation and divergence closures followed by classical
RK4:

```text
rhs = div(D * grad(s)) - div(v * interp(s))
```

This is not a five-point Laplacian or a lower-order substitute.  Scalar
boundaries remain fixed, the FP32 column-major field layout matches Formal,
and the RK4 accumulator is updated in the same `k1 + 2*k2 + 2*k3 + k4`
order.  Every CUDA variant evaluates the same face fluxes, divergences and
stage updates with the same operation order; the variants differ only in
how the work is mapped onto the GPU (fusion, tiling, on-chip reuse).

## Build and run

Requirements: an NVIDIA driver, a CUDA toolkit (`nvcc`) and a host C++
compiler supported by it.  The measurements used CUDA 13.3 with
`-arch=native` (sm_120).

```bash
bash benchmarks/2d-advection-cuda/run.sh                    # small CPU smoke run
FA_ALLOW_CUDA=1 GPU_LOCK=/path/to/lockfile \
  bash benchmarks/2d-advection-cuda/run.sh --mode cuda --input input.fadv2d
```

`run.sh` builds `build/formal_advection` with
`nvcc -O3 -std=c++17 -arch=native` and runs it.  `NVCC`, `CUDA_HOST_CXX`
(passed as `-ccbin`) and `CUDA_ARCH` override the toolchain.  CUDA mode is
opt-in (`FA_ALLOW_CUDA=1`); with `GPU_LOCK` set, CUDA runs go through
`flock "$GPU_LOCK"` so that timings on a shared GPU never overlap.

Options (same conventions as the Metal reference):

| option | meaning |
|---|---|
| `--mode cpu\|cuda` | CPU oracle or CUDA |
| `--input <file>` | FADV2D1 raw input (`mode raw`): grid, steps and FP32 `dt` come from its header |
| `--cells nx,ny`, `--steps n`, `--dt dt` | analytic Gaussian input instead (default `dt = 1e-4 * min(51/nx, 51/ny)^2`) |
| `--dump-bin [path]` | final scalar, headerless column-major FP32 `(nx+2, ny+2)` |
| `--dump [path]`, `--output path` | final scalar as CSV |
| `--kernel <variant>` | `auto` (default), `split`, `fused`, `tiled`, `prefetch`, `regy`, `regyf`, `pair`, `march`, `lean`; see [PERFORMANCE.md](./PERFORMANCE.md) |
| `--block x,y`, `--tile-x n`, `--tile-y n` | launch geometry (each variant has tuned defaults) |
| `--frame-parts n` | `march`: split boundary tiles into n row parts (default 2) |
| `--graph on\|off` | launch each timestep as a CUDA Graph (default on) |

The output line `RESULT mode=... solve_wall=... gpu_busy=...` reports the
solve-region wall time (all timesteps plus the final device-to-host copy of
the scalar; excluding initialization, upload, the untimed warm-up step and
file I/O) and the CUDA-event time of the timestep loop.  `--kernel auto`
selects `lean` (one kernel per stage) when the six full-size fields of the
per-stage scheme fit in the L2 cache and `march` (one fused kernel per
timestep) otherwise; the chosen variant is printed.  `march` needs `ny >= tile-y + 12` and falls back
to `regyf` on smaller grids.

## Validation

`validate.sh <case-dir> [options]` runs the native CPU oracle and native CUDA
on every `<case>.fadv2d` raw input of a directory and compares them, together
with the Formal final `<case>-formal.fadv2d`, using Formal's strict raw
comparator `example/scripts/compare_2d_raw_fields.py` (unchanged
`rtol=1e-6`, `atol=1e-8` at every point, exactly preserved boundaries,
`--evolution`):

```bash
PYTHON=/path/to/python GPU_LOCK=/path/to/lockfile \
  bash benchmarks/2d-advection-cuda/validate.sh cases/ --kernel regyf
```

Inputs and Formal finals come from the Formal example
(`--dump-input`, `--dump-final-raw`), e.g. for the benchmark case:

```bash
formal-native --solver workspace --order 4 --cells 2048,2048 --steps 1000 \
  --no-csv --dump-input cases/g2048.fadv2d --dump-final-raw cases/g2048-formal.fadv2d
```

`SKIP_CPU_ABOVE=<cells>` skips the (single-threaded) CPU oracle above that
many interior cells.  The validation record is in
[PERFORMANCE.md](./PERFORMANCE.md#validation).

## Provenance and licensing

`formal_advection.cu` ports the FP32 CPU oracle, the operator rows, the
command line and the raw formats of
[`../2d-advection-metal/formal_advection.m`](../2d-advection-metal/formal_advection.m)
and the kernel structure of
[`formal_advection.metal`](../2d-advection-metal/formal_advection.metal)
(the `split` variant is a direct port of the Metal kernels).  Those files
come from the LFortran repository at commit
`3d4c7941c5a153adb996ed62309225e871c06d9b`; the upstream LFortran BSD terms
are retained in [`LFortran-NOTICE.txt`](./LFortran-NOTICE.txt).  The CUDA
kernels, host driver and scripts are new.  Two deliberate host-side
differences: the working-set guard covers the CUDA buffers and is raised to
4 GiB (so 4096 x 4096 runs), and the warm-up restore re-uploads the
unchanged host fields instead of keeping backups.  The surrounding Formal
tree remains covered by [`LICENSE.txt`](../../LICENSE.txt).

Binaries, logs and raw fields belong under `build/` and are ignored.
