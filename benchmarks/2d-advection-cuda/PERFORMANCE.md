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
  excluded (the scope of the Metal reference's `solve_wall`).  The host
  array that receives the final copy is page-locked during initialization
  (all variants): from pageable memory the 16 MiB copy took 3.3 ms (5 ms
  from the padded layout) of the 2048² solve region, page-locked 0.6 ms.
  All timings in the results table are medians of 5 runs of the final source,
  interleaved over the variants of a size; the in-text A/B numbers were
  measured when each change was made.
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
  `smsp__inst_executed.sum`; Nsight Systems for per-kernel durations and
  overlap; PC-sampling stall reasons (`--section SourceCounters`) for the
  latency analysis.

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
at routine entry/exit, kernel time alone is ~0.53 s).  It was measured once,
before this work, and is not re-measured here.

### Theoretical peaks

| quantity | value | derivation |
|---|---:|---|
| FP32 | 116.95 TFLOP/s | 188 SMs x 128 FP32 lanes x 2 (FMA) x 2.430 GHz |
| DRAM (attribute formula) | 1.598 TB/s | 2 x 12481 MHz x 512 bit / 8 |
| DRAM (product specification, GDDR7 28 Gbps) | 1.792 TB/s | |
| DRAM, measured | 1.45 TB/s copy, 1.53 TB/s read | 2 GiB grid-stride kernels |
| L2, measured | 5.5 TB/s copy | 32 MiB grid-stride copy (L2 resident); `lean` sustains 6.0 TB/s |
| warp issue | 1.83 T warp-instructions/s | 188 SMs x 4 schedulers x 2.430 GHz |

The percentages below are relative to the attribute formula (1.598 TB/s)
unless stated.

## Results

Each row is a `--kernel` variant of the final source; all are retained and
validated (see [Validation](#validation)).  Solve wall in seconds for 1000
steps, medians of 5 interleaved runs:

| variant | 1024² | 2048² | 4096² | what changed |
|---|---:|---:|---:|---|
| LFortran Formal (`--gpu=cuda`) | 0.0935 | 0.5676 | n/a | baseline |
| `split` | 0.0739 | 0.5544 | 3.0221 | direct port of the Metal kernels: per stage a flux kernel writes four face arrays, an update kernel takes divergences and the RK update |
| `fused` | 0.0598 | 0.2023 | 1.4692 | one kernel per stage: scalar tile + halo and face fluxes in shared memory, divergence and RK update in the same kernel; no face arrays in global memory; stage 4 updates `base` only and stage 1 reads `base` |
| `tiled` | 0.0582 | 0.1692 | 1.4945 | warp-shaped 32-wide tiles, 32-bit indexing, branch-free interior operator rows for tiles away from boundaries |
| `prefetch` | 0.0575 | 0.1540 | 1.1582 | all global loads of a tile (velocities, base, accumulator) issued to registers before the first barrier |
| `regy` | 0.0591 | 0.1390 | 1.1658 | y direction register-blocked: a thread owns 3 consecutive cells of a column and computes their 6 y faces in registers; x faces via shared memory over a flat work list |
| `regyf` | 0.0387 | 0.1101 | 1.1497 | boundary tiles in a separate, concurrent frame kernel, split into 32 x 8 blocks; the interior kernel drops all general code (66 → 40 registers) |
| `pair` | 0.0569 | 0.1610 | 0.6499 | two RK stages per kernel with halo recomputation; stage-1/3 fields never leave the chip; half the DRAM traffic per step |
| `march` | 0.0914 | 0.1154 | **0.2969** | one kernel per timestep (all four stages): blocks march down column strips with a register pipeline in y; only base and the velocities are read and base written per step |
| `lean` | 0.0377 | 0.0833 | 1.1788 | one kernel per stage with the `march` layout on padded fields: every field read once, the stage input with a 3-row halo; boundary rim in a separate grid overlapped by programmatic dependent launch |
| `lean2` | **0.0274** | **0.0558** | 0.5851 | `lean` with two stages per kernel (1+2, 3+4): about 11 instead of 26 field accesses per step |
| `auto` (default) | 0.0274 | 0.0557 | 0.2971 | `lean2` if 6 fields fit in L2, else `march` |

Speed-up of the default over LFortran Formal: **10.2x at 2048²** and 3.4x at
1024².  Against the best variant before `march`/`lean`/`lean2` (`regyf` at
1024² and 2048², `pair` at 4096², same final source): 1.41x, 1.98x and
2.19x.  Against the compulsory-traffic roof (16 N² bytes per step at the
L2-resident copy rate of 5.5 TB/s for 1024² and 2048², at the 1.598 TB/s
DRAM peak for 4096²): 11%, 22% and **57%** (see the roofline table).
(The `march` and `lean2` rows were re-measured after the last `march`
change; the other rows come from the run before it.)  CUDA Graphs: the timestep loop has no host synchronization in any
mode, but each graph launch costs about a microsecond of GPU time; one
graph per timestep was even slower than direct launches for `lean2`
(0.0574 vs 0.0565 s).  Graphs of 10 timesteps (`--graph-steps`, default)
gain 2-5% over one graph per step for all variants (`lean2` 2048² 0.0574 →
0.0557, 1024² 0.0289 → 0.0273, `regyf` 2048² 0.1152 → 0.1104, `march`
4096² 0.341 → 0.335 s); 20 or 50 steps per graph gain nothing more.

### From `regyf` to `lean2` at 2048²

At 2048² the six live fields of the per-stage scheme (101 MB) stay in the
128 MB L2.  `regyf` moves 623 MB per step through L2 at 5.4 TB/s, about the
L2-resident copy rate, while issuing 114 M warp instructions per step (54%
of the issue slots): it is L2-bandwidth bound, with 1.3x halo over-fetch
over the 470 MB the per-stage scheme needs.  Two things had to shrink, the
bytes and the instructions.

`march` (below) cut both (309 MB, 50 M) but runs at only 24% issue
efficiency and ends up level with `regyf`.  Its layout and code, however,
are what the later variants are built from:

- a thread owns two adjacent columns; a block owns a strip of columns plus
  the stencil halo and marches down a segment of rows in chunks of 8, so
  the y direction is a register pipeline (each thread carries the last
  three stage values and y faces of its columns from chunk to chunk and
  evaluates one new y face per cell);
- the x direction goes through shared memory as float2: a row store, one
  barrier, each thread computes the x faces of its two columns, one
  barrier, each cell reads its four faces;
- the fields are kept in a padded device layout (32 ghost points of zeros
  on every side, even pitches), so no load needs a bounds check and a
  thread's column pair moves with one aligned 64-bit access;
- the exact factor 0.5 of the diffusive faces is folded into the gradient
  coefficients (bit-for-bit the same value, one multiplication less).

`lean` applies this to one kernel per stage.  A prologue loads the three
rows above the segment, so the stage input is read with only a 3-row halo
above and below and every other field (base, accumulator, both
velocities) exactly once.  Cells within 3-4 cells of a boundary (the rim)
need the general operator rows; they run as small shared-memory tiles in a
separate grid.  Getting that grid out of the way mattered most: launched on
side streams (fork/join), its blocks were scheduled after the interior grid
and added a 5-8 µs tail to every stage (0.094-0.104 s).  Launched first in
the same stream, with the interior grid launched by programmatic stream
serialization (it starts once all rim blocks have started and waits for the
rim grid only at its very end with `griddepcontrol.wait`), the two overlap:
0.0866 s.  Per step: 510 MB of L2 traffic at 6.0 TB/s and 45 M warp
instructions.  Alternatives for the rim were slower: per-cell general rows
straight from global memory (0.110 s), the edge strips doing the general
fix-up themselves (168-206 registers, 0.114 s), rim tiles merged into the
interior grid (higher registers, 0.109 s).

`lean` sits at the L2 roof of the per-stage scheme, so the remaining lever
is fewer passes.  `lean2` fuses the stages in pairs (1+2, 3+4).  A strip of
2*NT - 12 columns plus a 6-column halo marches down its segment with the
second stage three rows behind the first, both stages carrying their stage
values and y faces in registers; the first-stage values, rates and the
accumulator never leave the chip.  A prologue evaluates the first stage on
the six rows around the segment start, so beyond the 6-row halo of the
stage input nothing is loaded or computed twice.  The second stage takes
the velocities and base of its rows from registers (three rows carried
from the previous chunk, the rest from the first stage's loads): reloading
them missed L1 every time (0.0703 → 0.0635 s with the same 167 registers).
The rim (cells within 6-7 cells of a boundary: 32 x 6 tiles along the top
and bottom, 8 x 32 along the sides) evaluates both stages with the general
rows in shared memory, the first one on the tile extended by three cells;
the rim grids are also launched programmatically (each waits for the
previous strip grid, then releases the next), which hides part of the
launch gap (0.0582 → 0.0574 s).  Per step: 266 MB of L2 traffic, 46 M warp
instructions, 0.0574 s.

Sweeps (block threads / rows per segment, s): `lean` 2048² 96/32 0.0888,
64/32 0.0922, 64/48 0.0946, 128/32 0.106, 1024² 64/24 0.0408, 96/32
0.0446; `lean2` 2048² 64/64 0.0637, 128/64 0.0641, 64/32 0.0668, 96/32
0.0678, 64/80 0.0693, 1024² 64/24 0.0305, 96/24 0.0306, 64/16 0.0290 (after
the programmatic rim launch), 64/8 0.0379.  The defaults are 96/32 for
`lean` and 64/64 for `lean2` from 2 M cells on, 64/24 below.

Rejected for `lean2`: prefetching the next chunk's input rows and y-face
velocities into registers during the second stage (180 registers, one
block fewer per SM: 0.0582 → 0.0655 s); capping the registers at 128
(spills: 0.106 s); 64 x 6 / 8 x 64 rim tiles (0.059 s) and 16 x 6 / 8 x 16
rim tiles (0.079 s).  Without the rim grid at all (wrong boundaries) the
strips alone would take 0.0517 s, so the rim costs ~10%.

### `march`: temporal blocking over the whole timestep

`march` applies the same strip layout to all four stages in one kernel:
stage s runs 3s rows behind the newest input row, so per step global
memory sees one read of base and the velocities and one write of the new
base (base is double-buffered because neighbouring blocks read its halo).
A block of 128 threads owns a strip of 230 columns (plus the 4 x 3 = 12
column halo) and a segment of 64 rows; it starts 12 rows above its segment
(the four stages' warm-up) and skips stages whose rows are not valid yet.

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

At 4096² (DRAM resident) `march` moves 279-293 MB of DRAM traffic per step
(`pair` 747 MB, `lean2` 819 MB) and is 2.2x faster than `pair`.  It is
latency bound: 239-255 registers leave two warps per scheduler, and PC
sampling shows ~40% of the stall samples on long-scoreboard waits for the
per-stage reloads of base and the velocities (the per-SM working set of the
reloads, ~120 KB, does not fit L1).  The history of the kernel: one column
per thread, 123 registers, interior only 72 µs/step at 2048² (LSU bound,
1.4x warm-up overhead); two columns per thread 52 µs; the first frame
kernel took 420 µs because the compiler did not unroll its general code
(moved out of the unrolled loops into parallel fix-ups) and clamped every
load (replaced by the padded layout); issuing each stage's loads before its
first barrier and loading the next chunk in stage 4 gained 3% at 2048².
As in `lean2`, stages 2 and 4 now take the velocities and base of their
rows from stages 1 and 3 (three rows carried per column pair, the rest
from the current chunk) instead of reloading them; the interior kernel
stays at 255 registers with a 40-byte spill: 4096² 0.321 → 0.297 s at
128 threads / 128 rows per segment / 2 frame parts (the new default; the
segment height matters through the balance of frame and interior blocks:
64 rows 0.317, 96 0.316, 160 0.321, 256 0.303 s).
Rejected: `prefetch.global.L1` of the next chunk's rows (no gain), a
smaller shared-memory carveout for more L1 (no gain), forcing 168
registers (spills, slower), launching the frame grids first with
programmatic dependent launch (+5% at 2048², +7% at 4096²), splitting the
frame into more parts (best at 2-3).  Sweeps (2048², s): block 128 /
tile-y 64 0.119-0.121, 128/48/3 parts 0.119, 96/48 0.138; 4096² (200
steps): 128/64/2 0.0744, 128/48/1 0.0839, 64/128/2 0.0869.  A two-stage
version of the same kernel (stages 1+2 only, as a probe) took 64 µs per
launch at 2048², so with its 12-row warm-up and per-stage reloads the
`march` structure does not beat `lean2` at any fusion depth.

### Earlier steps

- `regyf` frame split: the general (boundary) path has about twice the
  latency of the interior one.  At 1024² the interior kernel of a stage
  takes only ~8 µs, and the frame kernel (146 tiles of 32 x 24, ~11 µs) was
  the critical path of every stage.  Splitting each frame tile into three
  blocks of 32 x 8 cells (`fa_regy_frame`) takes it off the critical path:
  1024² 0.0546 → 0.0430 s, 2048² unchanged.
- Geometry sweeps of the per-stage tile kernels (2048², median of 3, s):
  `fused` block 32x4 0.266, 32x8 0.208; `tiled`/`prefetch` 32x8 threads
  with 32x16 tiles best (0.175/0.158); `regy` 32x12 / tile 24 0.131 vs
  32x8/24 0.143 and 32x16/32 0.135; `regyf` 32x8 / tile 24 0.1207, 32x6/24
  0.1227, 32x8/32 0.1221, 32x12/24 0.1251, 32x8/16 0.1232, 32x8/48 0.1400
  (1024²: 32x8/16 0.0485, 32x12/24 0.0493, 32x8/24 0.0547 before the frame
  split); `pair` at 2048² tile 26x24 / 32x6 0.171-0.175 vs 58x12 0.179, at
  4096² (200 steps) tile 58x12 / block 64x6 0.1443 s best vs 26x24 0.1549.
- Rejected in commit `d353184`: a shared-memory-free "stream" kernel (warps
  walking 26-column strips with shuffles in x and a register pipeline in y;
  0.188 s, latency bound), a whole-timestep kernel with a 12-point halo
  ("step", 0.241 s: 4x face recomputation and 33% occupancy), and staging
  the y-face velocities in shared memory instead of registers (0.131 vs
  0.122 s).

## Roofline and peak percentages

Per timestep, from Nsight Compute (application replay) and the solve walls
of the results table.  "Useful" bandwidth counts only the compulsory
traffic of a step (read the scalar and both velocities, write the scalar
once: 4 field sizes, about 16 N² bytes, 67.2 MB at 2048²) divided by the
measured time; "% compulsory roof" relates it to the roof of the level the
data lives in: the measured L2-resident copy rate (5.5 TB/s) at 1024² and
2048², the DRAM attribute peak (1.598 TB/s) at 4096².  It is the fraction
of the time a kernel that moved only the compulsory bytes at that roof
would need.  "% L2 roof" is the L2 throughput over 5.5 TB/s, "issue" warp
instructions over the warp-issue peak.

| size | variant | ms/step | DRAM MB/step | DRAM TB/s | % DRAM peak | L2 MB/step | L2 TB/s | % L2 roof | L2 hit | GFLOP/step | TFLOP/s | % FP32 peak | issue | useful TB/s | % compulsory roof |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | LFortran Formal (initial) | 0.568 | 599 | 1.055 | 66.0% | 1691 | 2.98 | 54% | 83.9% | 1.213 | 2.14 | 1.8% |  | 0.118 | 2.2% |
| 2048 | `split` (initial) | 0.559 | 635 | 1.136 | 71.1% | 1430 | 2.56 | 47% | 78.2% | 1.515 | 2.71 | 2.3% |  | 0.120 | 2.2% |
| 2048 | `regyf` | 0.1101 | 52 | 0.472 | 29.6% | 623 | 5.66 | 103% | 100.0% | 1.420 | 12.90 | 11.0% | 57% | 0.611 | 11.1% |
| 2048 | `march` | 0.1155 | 5 | 0.043 | 2.7% | 309 | 2.68 | 49% | 100.0% | 1.609 | 13.93 | 11.9% | 24% | 0.582 | 10.6% |
| 2048 | `lean` | 0.0833 | 67 | 0.804 | 50.3% | 510 | 6.12 | 111% | 98.1% | 1.109 | 13.31 | 11.4% | 29% | 0.807 | 14.7% |
| 2048 | **`lean2` (default)** | 0.0557 | 32 | 0.575 | 36.0% | 266 | 4.78 | 87% | 99.8% | 1.269 | 22.78 | 19.5% | 45% | 1.207 | 21.9% |
| 1024 | LFortran Formal (initial) | 0.093 | 1 | 0.011 | 0.7% | 425 | 4.57 | 83% | 100.0% | 0.304 | 3.27 | 2.8% |  | 0.181 | 3.3% |
| 1024 | `regyf` | 0.0387 | 0 | 0.000 | 0.0% | 158 | 4.08 | 74% | 99.9% | 0.361 | 9.33 | 8.0% | 46% | 0.435 | 7.9% |
| 1024 | **`lean2` (default)** | 0.0274 | 0 | 0.000 | 0.0% | 79 | 2.88 | 52% | 99.9% | 0.353 | 12.88 | 11.0% | 30% | 0.615 | 11.2% |
| 4096 | `split` (initial) | 3.045 | 4015 | 1.319 | 82.5% | 5725 | 1.88 | 34% | 43.5% | 6.059 | 1.99 | 1.7% |  | 0.088 | 5.5% |
| 4096 | `regyf` (initial) | 1.16 | 1685 | 1.453 | 90.9% | 2485 | 2.14 | 39% | 42.2% | 5.620 | 4.84 | 4.1% |  | 0.232 | 14.5% |
| 4096 | `pair` | 0.6499 | 747 | 1.149 | 71.9% | 1992 | 3.07 | 56% | 60.2% | 7.694 | 11.84 | 10.1% | 45% | 0.413 | 25.9% |
| 4096 | `lean2` | 0.5851 | 819 | 1.400 | 87.6% | 1042 | 1.78 | 32% | 45.8% | 5.016 | 8.57 | 7.3% | 15% | 0.459 | 28.7% |
| 4096 | **`march` (default)** | 0.2969 | 293 | 0.987 | 61.8% | 866 | 2.92 | 53% | 76.2% | 5.532 | 18.63 | 15.9% | 30% | 0.905 | 56.6% |

Rows marked "initial" were profiled with the source of commit `96cd521`
(pageable final copy); their ms/step are from that measurement.  Relative to
the 1.792 TB/s specification the DRAM percentages are 0.89x these.  Nsight
Compute FLOP counts include the interior-formula faces of halos and are
counted per executed instruction (FFMA = 2).

Further experiments that did not pay off (final source unchanged):

- `lean2` with dataflow ordering instead of kernel boundaries: all four
  grids of a step launched programmatically and overlapping, each block
  waiting on completion counters of the 3 x 3 neighbouring blocks of the
  previous phase (invocation numbers from tickets, coherent loads).  A probe
  that simply let consecutive pair grids overlap (wrong, no waits) ran
  2048² in 0.038 s, so the kernel boundaries cost about a quarter of the
  time, but the correct version was slower: 0.0656 s (2048²), 0.0374 s
  (1024²); 0.0599 s with non-coherent loads (168 registers) - the waiting
  blocks hold the SM slots.  Its first version also showed why the
  invocation number cannot come from a per-block counter: a block of the
  next invocation can start before the same block of the previous one has
  finished.
- `lean2` loading the pair-1 accumulator rows with the other chunk loads
  (195 registers, 0.0557 → 0.0664 s).
- `lean2` rim tiles of other shapes at 1024² (16 x 6 / 8 x 16 0.0289 s,
  32 x 6 / 8 x 16 0.0276 s, 64 x 6 / 8 x 64 0.0383 s vs 0.0274 s); without
  the rim grid (wrong boundaries, serialized grids) 1024² would take
  0.0208 s and 2048² 0.0507 s, so the rim costs 25% and 9%.
- `lean2` with one-warp blocks (32 threads, 52 columns): 1024² 0.0273 s
  at 32 rows, no gain; 2048² 0.0613 s.
- `march` carrying the velocities and base across the stage-2/3 boundary
  as well (64-byte spill, 0.300 vs 0.297 s).

Why 2048² stays at 22% of the L2 roof: the `lean2` strip kernel issues 46 M
warp instructions per step (about 1.2x the FP32 operations the method needs
per cell and stage plus halo work), i.e. 25 µs at the full warp-issue
rate, and at 64 x 64 blocks the grid holds only ~6 resident warps per SM
(5.9 measured; more parallelism through shorter segments costs more halo
than it gains), so it issues on 43% of the cycles; its L2 traffic (266 MB)
already runs at 4.6 TB/s.  Reaching 50% of the compulsory roof would need
both all four stages on chip (the register budget of `march` shows the
cost) and near-full issue.

Which roof applies:

- **2048² is L2-resident.**  With the four face arrays gone (`fused` and
  later) the per-stage fields fit the 128 MB L2: DRAM sees only the
  write-back of dirty lines.  `regyf` is L2-bandwidth bound (5.4 TB/s), and
  so is `lean` with fewer bytes (6.0 TB/s, above the 5.5 TB/s copy
  benchmark).  `lean2` moves 2.3x fewer L2 bytes than `regyf` and is 2.0x
  faster; at 4.6 TB/s of L2 and 43% of the issue slots it is no longer
  purely bandwidth bound.  Its remaining time: the strip kernels (90%;
  167 registers leave two warps per scheduler, and the chunk loads at the
  start of each chunk are exposed) and the rim (~10%).  The useful
  bandwidth, 1.17 TB/s, is 73% of the DRAM peak although the data never
  leaves L2.
- **4096² is DRAM-resident.**  The per-stage scheme (`regyf`, `lean`) runs
  at the DRAM copy limit, so the only way forward is fewer bytes: `pair`
  and `lean2` halve them, `march` cuts them to 293 MB per step (1.09x the
  compulsory 269 MB).  `march` then runs at 62% of the DRAM peak (57% of
  the compulsory-traffic roof) and is latency bound (30% issue, 255
  registers); at the DRAM roof it would take ~0.18 ms/step.
- **FP32 is never the limit**: at most 22 TFLOP/s (19% of peak).  `lean2`
  performs 4.8 FLOP per byte of L2 traffic, against a machine balance of
  117 TFLOP/s / 5.5 TB/s = 21 FLOP/byte.
- **1024²** (4 MB per field) is latency/launch bound: 14.5 µs per pair
  kernel, with too few columns per segment row to fill the GPU; neither L2
  (2.7 TB/s) nor issue (29%) is saturated.

## Validation

Every variant (`split`, `fused`, `tiled`, `prefetch`, `regy`, `regyf`,
`pair`, `march`, `lean`, `lean2`, `auto`) of the final source passed the
strict raw comparator (`rtol=1e-6`, `atol=1e-8` at every point, exact fixed
boundaries, `--evolution`) against both the native CPU oracle and the
LFortran-compiled Formal final on:

- small cases: Gaussian 17x13/9, 51x51/500, 130x67/50, 257x257/200 and
  nonzero, non-smooth-boundary random-velocity cases 13x11/3, 37x23/7,
  70x45/20 (inputs from `ci/cuda_nvidia/make_boundary_input.py` of the
  LFortran CUDA recipe);
- medium cases: Gaussian 1000x700/20 and boundary 613x451/10;
- Gaussian 1024x1024/1000, 2048x2048/20 and the full **2048x2048/1000**
  benchmark case (CPU oracle run for the full 1000 steps);
- nonzero, non-smooth-boundary random-velocity inputs 61x83, 64x64, 130x97,
  333x251, 517x1031 and 1100x300 (10 steps, from the same script; compared
  against the CPU oracle with the same strict comparator).

`march`, `lean` and `lean2` were also run with several geometries
(`--block 64/96/128`, `--tile-y 8..64`, `march --frame-parts 1..3`,
`--graph off`) on these inputs; `march` falls back to `regyf` when
`ny < tile-y + 12`.  `compute-sanitizer` memcheck reported no errors for
any variant on 13x11, 37x23, 130x67, 257x257, 1000x999, 61x1000 and
4096x4096, nor for `march`, `lean` and `lean2` on 9x9, 61x83, 130x97,
517x1031, 1100x300, 1000x61 and 1000x999; racecheck and synccheck reported
no hazards for them on 130x97.  At 4096² (no Formal or CPU final available)
`regyf`, `pair` and `march` agree with `split` within the same tolerances
(max abs diff 4.2e-7, relative L2 4.9e-8 to 5.5e-8).

Comparator output for the default kernel (`lean2`) on the shared
2048²/1000 input (LFortran final as the Formal final):

```text
cells=2048,2048 steps=1000 dt=6.2012674106881605e-08 elements=4202500
Formal evolution:         max_abs_diff=4.1723251342773438e-04 relative_l2=2.8551724594974868e-04
native evolution:         max_abs_diff=4.1723251342773438e-04 relative_l2=2.8551945259266812e-04
CPU oracle evolution:     max_abs_diff=4.1723251342773438e-04 relative_l2=2.8552615593000581e-04
Formal versus native:     max_abs_diff=2.3841857910156250e-07 max_rel_diff=2.8559188919034697e-04 relative_l2=4.3890333915673501e-08
CPU oracle versus Formal: max_abs_diff=4.7683715820312500e-07 max_rel_diff=3.1969309462915604e-04 relative_l2=8.0657250227567590e-08
CPU oracle versus native: max_abs_diff=3.5762786865234375e-07 max_rel_diff=1.6417665407978985e-04 relative_l2=5.4726786541499033e-08
full-field comparison passed; fixed boundaries preserved
```

(`regyf`: Formal versus native 2.38e-7 / relative L2 4.45e-8; `lean`
2.98e-7 / 4.37e-8; `march` 2.38e-7 / 4.31e-8.)

## Kernel structure for a code generator

What a compiler has to produce to match `lean2`, the default for
L2-resident problems, in order of measured impact:

1. **No face temporaries in global memory** (0.559 → 0.201 s).  Fuse
   gradient, interpolation, face flux, divergence and the stage update;
   face fluxes are produced and consumed on chip.  This cuts the live
   footprint from 10 to 6 field-sized arrays (fits L2 at 2048²).  The last
   stage writes only `base`, and the next step's first stage reads `base`
   directly; fixed boundary points are never written.
2. **Two RK stages per kernel** (`lean` 0.085 → `lean2` 0.057 s): the
   second stage of a pair runs three rows behind the first inside the same
   block; the first-stage values, rates and the accumulator stay on chip.
   `base` is double-buffered between timesteps because neighbouring blocks
   read its halo.  (All four stages in one kernel, `march`, is the right
   choice once the fields no longer fit L2.)
3. **Strips that march in y.**  A thread owns two adjacent columns; a
   block owns 2*NT - 12 columns plus a 6-column halo and walks down a
   segment of rows in chunks of 8.  The y direction is a register pipeline
   (three stage values and three y faces carried per stage and column, one
   new y face per cell); the x direction goes through shared memory as
   float2 with two barriers per stage and chunk.  A prologue covers the
   rows above the segment, so nothing beyond the stencil halo is loaded or
   computed twice.
4. **Load once, carry in registers.**  All global loads of a chunk are
   issued at its start; the second stage takes the velocities and base of
   its rows from the first stage's loads plus three rows carried from the
   previous chunk (0.0703 → 0.0635 s).  Use an aligned, padded device
   layout so loads are unchecked 64-bit accesses.
5. **Specialize away the boundaries, and keep them off the critical path.**
   The strips use only the interior operator rows.  The rim of cells whose
   cones reach a boundary runs as small tiles with the general rows in a
   separate grid, launched first in the same stream; the strip grid is
   launched with programmatic dependent launch so the two overlap (side
   streams with fork/join events left a 5-8 µs tail per stage).
6. **Synchronization:** no grid-wide synchronization, no host
   synchronization inside the timestep loop; launch order on one stream
   (plus `griddepcontrol.wait` at the end of programmatically launched
   grids) provides all dependencies.  CUDA Graphs are optional.
7. **Occupancy is the binding resource.**  The strip kernels are limited
   by registers (167 for `lean2`, 239-255 for `march`); prefetching more
   data or capping registers both lost.  Block shape and segment height
   matter through wave quantization (64 threads x 64 rows at 2048²).
