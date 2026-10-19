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
  5 runs, interleaved per size.  Since `lean2` the host array that receives
  the final copy is page-locked during initialization (all variants): the
  pageable copy of 16 MiB took 3.3 ms (5 ms from the padded layout) of the
  2048² solve region, the page-locked one 0.6 ms.  Rows measured before
  that change are marked in the tables below.
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
| `regyf` | 0.0430 | **0.1164** | 1.1603 | boundary tiles moved to a separate, concurrent frame kernel; the interior kernel drops all general code (66 → 40 registers); frame tiles split into 32 x 8 blocks (1024²: 0.0546 → 0.0430) |
| `pair` | 0.0774 | 0.1786 | 0.6745 | two RK stages per kernel with halo recomputation; stage-1/3 fields never leave the chip; half the DRAM traffic per step |
| `march` | 0.0977 | 0.1209 | **0.3590** | one kernel per timestep (all four stages): blocks march down column strips with a register pipeline in y; only base and the velocities are read and base written per step (see below) |
| `lean` | 0.0399 | 0.0866 | 1.1984 | one kernel per stage with the `march` thread layout and padded fields: every field read once (plus a 3-row halo of the stage input), boundary rim in a concurrent kernel overlapped by programmatic dependent launch (see below) |
| `lean2` | **0.0302** | **0.0635** | 0.6320 | `lean` with two stages per kernel (1+2, 3+4): the stage-1/3 values, rates and accumulator stay on chip, about 11 instead of 26 field accesses per step (see below) |
| `auto` (default) | 0.0302 | 0.0635 | 0.3590 | `lean2` if 6 fields fit in L2, else `march` |

Speed-up of the default over LFortran Formal: **8.9x at 2048²** and 3.1x at
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

### `march`: temporal blocking over the whole timestep

A block of 128 threads owns a strip of 230 columns (each thread two adjacent
columns, plus the 4 x 3 = 12-column halo of four stages on each side) and a
segment of 64 rows.  It marches down the segment in chunks of 8 rows,
starting 12 rows above it: per chunk it loads 8 new rows of base and pushes
them through all four stages, stage s producing the rows 3s behind the
newest input.  Each thread carries, per stage and column, the last three
stage values, y faces and accumulator values from chunk to chunk in
registers, so the y direction costs one new face per cell and stage; the x
faces go through shared memory (float2 per thread, one face computed per
face).  The stage values, rates and the accumulator never leave the chip:
global memory sees one read of base and the velocities and one write of the
new base per step (base is double-buffered because neighbouring blocks read
its halo).  The fields live in a padded device layout (32 ghost points,
zero, on every side; even pitches), so no load needs a bounds check and a
thread's column pair moves with one 64-bit access.

Blocks whose dependency cone reaches a boundary run in three concurrent
frame kernels.  x boundaries are handled per thread (the boundary-side faces
and the x parts of cells 1 and nx are recomputed with the general rows by
the block in parallel).  In y the one-sided rows reach five cells, beyond
the 3-row lag of the pipeline: the top blocks start their march at row -14
and the bottom blocks at a row congruent to ny - 1 (mod 8), which places
cells 1..3 (ny-2..ny) of every stage into one chunk whose register window
holds all seven rows their general rows need (at the bottom up to three rows
are kept from the previous chunk); those cells take their y part from the
general rows.  Frame tiles are split into `--frame-parts` (default 2) row
parts so that they finish with the interior blocks.

At 4096² `march` moves 278 MB of DRAM traffic per step (`pair`: 747 MB) and
is 1.92x faster than `pair` (interleaved, 5 runs each: 0.345-0.360 s vs
0.683-0.695 s).  At 2048² it executes 48 M warp instructions and 303 MB of
L2 traffic per step against 110 M and 619 MB for `regyf`, but it is latency
bound (about 2 warps per scheduler at 239-255 registers, with long
global-load and barrier waits): it issues on only ~21% of the cycles and
ends up level with `regyf`.  At 1024² there are too few columns to fill the
GPU without very short segments, whose 12-row warm-up dominates.  Sweeps
(2048², s): block 128 / tile-y 48 / 3 parts 0.119, 96/4 parts 0.119,
32/2 0.122, 64/2 0.124; block 64 0.128-0.153; 4096² (200 steps): 128/64/2
0.0744, 128/48/1 0.0839, 64/128/2 0.0869.

`regyf` frame split: the general (boundary) path has about twice the
latency of the interior one.  At 1024² the interior kernel of a stage takes
only ~8 µs, and the frame kernel (146 tiles of 32 x 24, ~11 µs) was the
critical path of every stage.  Splitting each frame tile into three blocks
of 32 x 8 cells (`fa_regy_frame`) takes it off the critical path: 1024²
0.0546 → 0.0430 s (interleaved, 5 runs each), 2048² unchanged.

### `lean`: a per-stage kernel at the L2 roof

At 2048² everything stays in L2, and `regyf` is bound by L2 bandwidth
(619 MB per step at ~5.3 TB/s) while issuing 110 M warp instructions per
step; `march` cuts both (303 MB, 48 M) but is latency bound.  `lean` keeps
one kernel per stage and takes the lean code of `march`:

- Strips of 2*NT - 6 columns (a thread owns two adjacent columns, 3-column
  halo on each side) march down segments of H rows in chunks of 8 rows.  A
  prologue loads the three rows above the segment, so the stage input is
  read with only a 3-row halo above and below and every other field (base,
  accumulator, both velocities) exactly once, all with aligned 64-bit loads
  from the padded layout of `march`; the y direction stays in registers,
  the x direction goes through shared memory as float2.
- The interior rectangle (columns [5, nx-2), rows [4, ny-2)) needs only the
  interior operator rows.  The rim of 3-4 cells along the boundaries runs as
  small tiles with the general rows (64 x 3 along the top and bottom, 4 x 48
  along the sides) in a separate grid launched first in the same stream; the
  interior grid is launched with programmatic stream serialization, so it
  starts as soon as all rim blocks have started (they get the SMs first and
  the grids overlap), and it waits for the rim grid only at its very end
  (`griddepcontrol.wait`), which orders the next stage after both.
- base is double-buffered (stage 4 writes the other buffer).

Per step at 2048²: 510 MB of L2 traffic (5.9 TB/s), 44.6 M warp
instructions, 66 MB DRAM write-back; 0.0866 s for 1000 steps (`regyf`
0.1166 s, interleaved, 5 runs each).  The rim placement mattered: rim tiles
in two side streams (fork/join) were scheduled after the interior grid and
added a 5-8 µs tail per stage (0.094-0.104 s); rim cells computed per thread
straight from global memory (0.110 s) or by the edge strips themselves (with
the general code in the same kernel: 168-206 registers, 0.114 s) were
slower; merging the rim tiles into the interior grid raised its registers
(0.109 s).  Sweep at 2048² (block threads / rows per segment, s): 96/32
0.0888, 64/32 0.0922, 64/48 0.0946, 96/48 0.0958, 96/40 0.0959, 128/32
0.106; 1024²: 64/24 0.0408, 96/32 0.0446.

### `lean2`: two stages per kernel

`lean` sits at the L2 roof of the per-stage scheme, so the next step is
fewer passes.  `lean2` fuses the stages in pairs like `pair`, but with the
`lean` structure: a strip (2*NT - 12 columns plus a 6-column halo on each
side) marches down its segment with the second stage three rows behind the
first, both stages carrying their stage values and y faces in registers; a
prologue evaluates the first stage on the six rows around the segment start,
so beyond the 6-row halo of the stage input nothing is loaded or computed
twice.  The rim (cells within 6-7 cells of a boundary: 32 x 6 tiles along
the top and bottom, 8 x 32 tiles along the sides) evaluates both stages with
the general rows in shared memory (the first one on the tile extended by
three cells), in a grid launched before the strips with the same
programmatic-launch overlap as `lean`.  Per step at 2048²: 310 MB of L2
traffic (`lean` 510 MB, `regyf` 619 MB), 46 M warp instructions, 34 MB DRAM
write-back; 0.0682 s for 1000 steps (`lean` 0.0892 s, interleaved, 5 runs
each).  166-168 registers; the strip kernel issues on ~37% of the cycles
and runs L2 at ~4.5 TB/s, so it is neither purely bandwidth nor purely
latency bound.  The second stage takes the velocities and base of its rows
from registers (three rows carried from the previous chunk, the rest from
the first stage's loads) instead of reloading them, which the L1 did not
catch: 2048² 0.0703 → 0.0635 s, 1024² 0.0323 → 0.0302 s (interleaved),
same register count.  Sweeps (block threads / rows per segment, s): 2048²
64/64 0.0637, 128/64 0.0641, 64/32 0.0668, 96/32 0.0678, 64/80 0.0693;
1024² 64/24 0.0305, 96/24 0.0306, 64/32 0.0325.

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
`pair`, `march`, `auto`) passed the strict raw comparator (`rtol=1e-6`, `atol=1e-8`
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
13x11, 37x23, 130x67, 257x257, 1000x999, 61x1000 and 4096x4096.

`march` (whose boundary handling only runs when `ny >= tile-y + 12`; the
smallest cases above fall back to `regyf`) was additionally validated with
several geometries (`--block 64/96/128`, `--tile-y 24..64`,
`--frame-parts 1..3`) on nonzero, non-smooth-boundary random-velocity inputs
61x83, 64x64, 130x97, 333x251, 517x1031 and 1100x300 (10 steps, written by
the LFortran recipe's `make_boundary_input.py`; compared against the CPU
oracle with the same strict comparator), on all the cases above, and under
`compute-sanitizer` memcheck, racecheck and synccheck.  At 4096² it agrees
with `split` within max abs diff 4.2e-7, relative L2 5.4e-8.  At
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
