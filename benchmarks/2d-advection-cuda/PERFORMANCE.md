# Native C + CUDA performance notes

The native program is the numerical oracle and the NVIDIA performance target
for Formal's corrected 2D advection-diffusion workspace: FP32 order-4 mimetic
gradient, interpolation and divergence with fixed boundaries and classical
RK4 (`k1 + 2*k2 + 2*k3 + k4`).  All optimizations below keep that method and
operation order; none substitutes a five-point Laplacian or changes the
answer beyond FP32 rounding of fused multiply-adds.

## Method

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs,
  128 MB L2, 96 GB GDDR7, max SM clock 2430 MHz, memory clock 12481 MHz,
  512-bit bus), driver 610.43.02, CUDA 13.3, `nvcc -O3 -arch=native`.
- Timing: `solve_wall` = wall time of all timesteps plus the final
  device-to-host copy of the scalar, after an untimed warm-up step and a
  re-upload of the initial state; initialization, upload and file I/O are
  excluded (the scope of the Metal reference's `solve_wall`).  Medians of
  5 runs, interleaved per size.
- The GPU was shared.  Every timed or profiled run was serialized with
  `flock <lockfile>`, and each timed run additionally waited until
  `nvidia-smi` showed no other compute process.  Unrelated jobs outside the
  lock occasionally still overlapped and inflated individual samples (up to
  ~30%); decisions between close variants used interleaved A/B runs.
- Inputs: the 2048² case uses the shared Formal input (SHA-256
  `5021f0823ff2d289be9d947c37a463d69be67ea0ce2c56784d145da8278e6980`, Formal
  `--cells 2048,2048 --steps 1000`, `dt = 6.20126741e-08`), the 1024² case
  the corresponding Formal input; 4096² uses the native analytic Gaussian
  (`--cells 4096,4096 --steps 1000`) because the Formal example rejects it
  (its 512 MiB workspace guard).
- Profiling: Nsight Compute 2026 with `--replay-mode application
  --cache-control none --clock-control none` (application replay keeps the
  real L2 state between kernels; kernel replay with memory save/restore
  pollutes L2 and overstates DRAM traffic), two timesteps after the warm-up,
  metrics `dram__bytes_op_{read,write}.sum`, `lts__t_bytes.sum`,
  `lts__t_sector_hit_rate.pct`,
  `sm__sass_thread_inst_executed_op_{ffma,fadd,fmul}_pred_on.sum`,
  `smsp__inst_executed.sum`; Nsight Systems for per-kernel durations.

Commands (from this directory, after `bash run.sh`; `L` is the lock file):

```bash
flock "$L" build/formal_advection --mode cuda --input g2048.fadv2d --kernel <variant>
flock "$L" build/formal_advection --mode cuda --cells 4096,4096 --steps 1000 --kernel <variant>
flock "$L" ncu --replay-mode application --cache-control none --clock-control none \
  --kernel-name regex:fa_ --launch-skip <3 steps of kernels> --launch-count <2 steps> \
  --csv --page raw --metrics <list above> \
  build/formal_advection --mode cuda --input g2048.fadv2d --graph off --kernel <variant>
```

The LFortran baseline is the Formal example built with LFortran
`--gpu=cuda` (`--solver workspace --order 4 --input g2048.fadv2d --no-csv`,
`solve_seconds`; its solve region also contains ~39 ms of host/device copies
at routine entry/exit, kernel time alone is ~0.53 s).

### Theoretical peaks

| quantity | value | derivation |
|---|---:|---|
| FP32 | 116.95 TFLOP/s | 188 SMs x 128 FP32 lanes x 2 (FMA) x 2.430 GHz |
| DRAM (attribute formula) | 1.598 TB/s | 2 x 12481 MHz x 512 bit / 8 |
| DRAM (product specification, GDDR7 28 Gbps) | 1.792 TB/s | |
| DRAM, measured | 1.45 TB/s copy, 1.53 TB/s read | 2 GiB grid-stride kernels |
| L2, measured | 5.5 TB/s copy | 32 MiB grid-stride copy (L2 resident) |

The percentages below are relative to the attribute formula (1.598 TB/s)
unless stated.

## Optimization log

Each row is a `--kernel` variant of the final source; all are retained and
validated (see [Validation](#validation)).  Solve-wall medians of 5 runs:

| variant | 1024² | 2048² | 4096² | what changed |
|---|---:|---:|---:|---|
| LFortran Formal (`--gpu=cuda`) | 0.0935 | 0.5676 | n/a | baseline |
| `split` | 0.0768 | 0.5586 | 3.0451 | direct port of the Metal kernels: per stage a flux kernel writes four face arrays, an update kernel takes divergences and the RK update |
| `fused` | 0.0620 | 0.2008 | 1.4846 | one kernel per stage: scalar tile + halo and face fluxes in shared memory, divergence and RK update in the same kernel; no face arrays in global memory; stage 4 updates `base` only and stage 1 reads `base` |
| `tiled` | 0.0597 | 0.1683 | 1.5083 | warp-shaped 32-wide tiles, 32-bit indexing, branch-free interior operator rows for tiles away from boundaries |
| `prefetch` | 0.0598 | 0.1538 | 1.1725 | all global loads of a tile (velocities, base, accumulator) issued to registers before the first barrier |
| `regy` | 0.0615 | 0.1400 | 1.1799 | y direction register-blocked: a thread owns 3 consecutive cells of a column and computes their 6 y faces in registers; x faces via shared memory over a flat work list |
| `regyf` | 0.0544 | **0.1164** | 1.1603 | boundary tiles moved to a separate, concurrent frame kernel; the interior kernel drops all general code (66 → 40 registers) |
| `pair` | 0.0774 | 0.1786 | **0.6745** | two RK stages per kernel with halo recomputation; stage-1/3 fields never leave the chip; half the DRAM traffic per step |
| `auto` (default) | 0.0545 | 0.1180 | 0.6806 | `regyf` if 6 fields fit in L2, else `pair` |

Speed-up of the default over LFortran Formal: **4.8x at 2048²** and 1.7x at
1024².  CUDA Graphs (one graph per timestep) made no measurable difference
(`split` 0.5583 vs 0.5590 s, `fused` 0.2083 vs 0.2095 s, graph off vs on):
launches are asynchronous and there is no host synchronization inside the
timestep loop in either mode.

Geometry sweeps (2048², median of 3, s): `fused` block 32x4 0.266, 32x8
0.208; `tiled`/`prefetch` 32x8 threads with 32x16 tiles best (0.175/0.158);
`regy` 32x12 / tile 24 0.131 vs 32x8/24 0.143 and 32x16/32 0.135 in its
original sweep; `regyf` 32x8 / tile 24 0.1207, 32x6/24 0.1227, 32x8/32
0.1221, 32x12/24 0.1251, 32x8/16 0.1232, 32x8/48 0.1400; `pair` at 2048²
tile 26x24 / 32x6 0.171-0.175 vs 58x12 0.179, at 4096² (200 steps) tile
58x12 / block 64x6 0.1443 s best vs 26x24 0.1549.

Rejected experiments (in commit `d353184`): a shared-memory-free "stream"
kernel (warps walking 26-column strips with shuffles in x and a register
pipeline in y; 0.188 s, latency bound), a whole-timestep kernel with a
12-point halo ("step", 0.241 s: 4x face recomputation and 33% occupancy),
and staging the y-face velocities in shared memory instead of registers
(slower in an interleaved A/B: 0.131 vs 0.122 s).

## Roofline and peak percentages

Per timestep, from Nsight Compute (application replay) and the measured
solve walls.  "Useful" bandwidth counts only the compulsory traffic of a
step (read the scalar and both velocities, write the scalar once: 4 x 16.8 MB
at 2048²) divided by the measured time.

| size | variant | ms/step | DRAM MB/step | DRAM TB/s | % DRAM peak | L2 MB/step | L2 TB/s | L2 hit | GFLOP/step | TFLOP/s | % FP32 peak | useful TB/s | useful % DRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | LFortran Formal | 0.568 | 599 | 1.056 | 66.1% | 1691 | 2.98 | 83.9% | 1.213 | 2.14 | 1.8% | 0.118 | 7.4% |
| 2048 | `split` | 0.559 | 635 | 1.137 | 71.2% | 1430 | 2.56 | 78.2% | 1.515 | 2.71 | 2.3% | 0.120 | 7.5% |
| 2048 | `fused` | 0.201 | 54 | 0.270 | 16.9% | 643 | 3.20 | 100.0% | 1.603 | 7.98 | 6.8% | 0.335 | 21.0% |
| 2048 | `tiled` | 0.168 | 55 | 0.326 | 20.4% | 642 | 3.82 | 100.0% | 1.290 | 7.67 | 6.6% | 0.400 | 25.0% |
| 2048 | `prefetch` | 0.154 | 50 | 0.326 | 20.4% | 639 | 4.16 | 100.0% | 1.255 | 8.16 | 7.0% | 0.437 | 27.4% |
| 2048 | `regy` | 0.140 | 50 | 0.359 | 22.5% | 629 | 4.49 | 100.0% | 1.410 | 10.07 | 8.6% | 0.480 | 30.1% |
| 2048 | `regyf` (default) | 0.116 | 51 | 0.441 | 27.6% | 619 | 5.32 | 99.9% | 1.410 | 12.12 | 10.4% | 0.578 | 36.2% |
| 2048 | `pair` | 0.179 | 34 | 0.192 | 12.0% | 505 | 2.83 | 99.8% | 1.956 | 10.95 | 9.4% | 0.376 | 23.6% |
| 1024 | LFortran Formal | 0.093 | 1 | 0.006 | 0.4% | 425 | 4.55 | 100.0% | 0.304 | 3.25 | 2.8% | 0.180 | 11.3% |
| 1024 | `regyf` (default) | 0.054 | 0 | 0.000 | 0.0% | 155 | 2.85 | 99.8% | 0.355 | 6.54 | 5.6% | 0.310 | 19.4% |
| 4096 | `split` | 3.045 | 4015 | 1.318 | 82.5% | 5725 | 1.88 | 43.5% | 6.059 | 1.99 | 1.7% | 0.088 | 5.5% |
| 4096 | `regyf` | 1.160 | 1685 | 1.452 | 90.9% | 2485 | 2.14 | 42.2% | 5.620 | 4.84 | 4.1% | 0.232 | 14.5% |
| 4096 | `pair` (default) | 0.675 | 747 | 1.107 | 69.3% | 1992 | 2.95 | 60.2% | 7.694 | 11.41 | 9.8% | 0.398 | 24.9% |

Relative to the 1.792 TB/s specification the DRAM percentages are 0.89x
these (e.g. `regyf` at 4096²: 81.0%; LFortran at 2048²: 58.9%).  Nsight
Compute FLOP counts include the interior-formula faces of halos and are
counted per executed instruction (FFMA = 2).

Which roof applies:

- **2048² is L2-resident.**  With the four face arrays gone (`fused` and
  later), the six live fields (base, two stage buffers, accumulator, two
  velocities: 101 MB) fit the 128 MB L2: DRAM reads drop to zero and only
  ~50 MB/step of dirty lines are written back.  The default `regyf` moves
  5.3 TB/s through L2, at the measured L2-resident copy rate (5.5 TB/s):
  it is **L2-bandwidth bound**.  Its 619 MB/step of L2 traffic is 1.3x the
  per-stage minimum (about 470 MB: per stage read 5 fields, write 2), the
  rest being halo over-fetch.  The LFortran build (and `split`) keeps four
  extra face arrays (168 MB) and is **DRAM bound** at 66% (71%) of peak with
  only 7.4% useful bandwidth.
- **4096² is DRAM-resident.**  `regyf` reaches 1.45 TB/s (91% of the
  attribute peak, at the measured copy limit), so the only way forward is
  fewer bytes: `pair` moves 2.3x fewer DRAM bytes per step and is 1.72x
  faster although it runs at 69% of peak (it is issue/latency limited by
  its larger shared-memory footprint and halo recomputation).
- **FP32 is never the limit**: at most 12 TFLOP/s (10% of peak).  The
  default kernel performs 2.3 FLOP per byte of L2 traffic, against a
  machine balance of 117 TFLOP/s / 5.5 TB/s = 21 FLOP/byte.
- **1024²** (4 MB per field) is latency/launch bound: 13.6 µs per stage
  with ~5 µs kernels of only ~1400 blocks; neither L2 (2.9 TB/s) nor issue
  is saturated.

The remaining gap at 2048² is the halo over-fetch (scalar tile 40 x 32 for
32 x 24 cells, `(RY+3)/RY` y-face velocities) and the global round trips of
the stage buffers and accumulator; fusing stages (`pair`) removes the latter
but costs more instructions than it saves while the data is L2 resident.

## Validation

Every variant (`split`, `fused`, `tiled`, `prefetch`, `regy`, `regyf`,
`pair`, `auto`) passed the strict raw comparator (`rtol=1e-6`, `atol=1e-8`
at every point, exact fixed boundaries, `--evolution`) against both the
native CPU oracle and the LFortran-compiled Formal final on:

- small cases: Gaussian 17x13/9, 51x51/500, 130x67/50, 257x257/200 and
  nonzero, non-smooth-boundary random-velocity cases 13x11/3, 37x23/7,
  70x45/20 (inputs from `ci/cuda_nvidia/make_boundary_input.py` of the
  LFortran CUDA recipe);
- medium cases: Gaussian 1000x700/20 and boundary 613x451/10;
- Gaussian 2048x2048/20 and the full **2048x2048/1000** benchmark case
  (CPU oracle run for the full 1000 steps).

`compute-sanitizer` (memcheck) reported no errors for any variant on
13x11, 37x23, 130x67, 257x257, 1000x999, 61x1000 and 4096x4096.  At
4096² (no Formal or CPU final available) `regyf` and `pair` agree with
`split` within the same tolerances (max abs diff 4.2e-7, relative L2
4.9e-8 / 5.5e-8).

Comparator output for the default kernel on the shared 2048²/1000 input
(`validate.sh` with the LFortran final as `<case>-formal.fadv2d`):

```text
cells=2048,2048 steps=1000 dt=6.2012674106881605e-08 elements=4202500
Formal evolution:      max_abs_diff=4.1723251342773438e-04 relative_l2=2.8551724594974868e-04
native evolution:      max_abs_diff=4.1723251342773438e-04 relative_l2=2.8552066010787769e-04
CPU oracle evolution:  max_abs_diff=4.1723251342773438e-04 relative_l2=2.8552615593000581e-04
Formal versus native:  max_abs_diff=2.3841857910156250e-07 max_rel_diff=3.1969309462915604e-04 relative_l2=4.4532275363863281e-08
CPU oracle versus Formal: max_abs_diff=4.7683715820312500e-07 max_rel_diff=3.1969309462915604e-04 relative_l2=8.0657250227567590e-08
CPU oracle versus native: max_abs_diff=3.5762786865234375e-07 max_rel_diff=1.6345210853220007e-04 relative_l2=5.3000014208420572e-08
full-field comparison passed; fixed boundaries preserved
```

(`pair` on the same input: Formal versus native 2.98e-7 / relative L2
4.69e-8; CPU oracle versus native 2.98e-7 / 5.01e-8.)

## Kernel structure for a code generator

What a compiler has to produce to match `regyf`, the default for
L2-resident problems, in order of measured impact:

1. **One kernel per RK stage, no face temporaries in global memory**
   (0.559 → 0.201 s).  Fuse gradient, interpolation, face flux, divergence
   and the stage update.  Face fluxes are produced inside the kernel and
   consumed from shared memory/registers.  This cuts the live footprint from
   10 to 6 field-sized arrays (fits L2 at 2048²) and DRAM traffic per step
   from 635 MB to ~50 MB.  The last stage writes only `base`, and the next
   step's first stage reads `base` directly (no copy kernel); fixed boundary
   points are never written.
2. **Warp = 32 consecutive x cells** (column-major, x fastest), block
   32 x 8 threads, tile 32 x 24 cells.  The block loads the scalar tile with
   a 5-point halo (40 x 32 floats; 5 covers the one-sided boundary rows) into
   shared memory with coalesced row loads, using 32-bit index arithmetic.
3. **Issue all global loads first.**  Velocities for the tile's faces, base
   and accumulator values are loaded into registers before the first
   `__syncthreads`, so their latency overlaps the tile load (0.168 → 0.154).
4. **x faces in shared memory, y faces in registers.**  The tile's x faces
   (24 rows x 37 faces, diffusive and advective) are computed once, each
   thread taking items of a flat list, and stored to shared memory; after
   one barrier each cell reads its 4 faces per kind.  Each thread owns 3
   consecutive cells of one column and computes their 6 y faces from 9
   shared-memory scalars into registers (no shared y-face arrays) (→ 0.140).
5. **Specialize away the boundaries.**  Interior tiles use only the interior
   stencil rows, branch free.  Tiles within reach of a boundary run in a
   second kernel launched concurrently on another stream (fork/join events,
   captured in the timestep's CUDA Graph); they compute faces with the
   interior rows and take the x or y part of a boundary cell's divergence
   from the general rows.  Keeping the general code out of the interior
   kernel lowers it from 66 to 40 registers and raises occupancy to 100%
   (→ 0.116).
6. **Synchronization:** two `__syncthreads` per stage kernel, no grid-wide
   synchronization, no host synchronization inside the timestep loop
   (launch order on one stream provides the stage dependencies).  CUDA
   Graphs are optional.
7. **For working sets beyond L2** fuse two stages per kernel (`pair`): the
   block computes the first stage on its tile extended by the stencil
   radius 3 into shared memory and the second stage on the tile, keeping
   the stage-1/3 values and rates on chip; `base` is double-buffered
   between timesteps because neighbouring tiles read its halo.  This halves
   DRAM traffic (1685 → 747 MB/step at 4096²) and is 1.72x faster there.
