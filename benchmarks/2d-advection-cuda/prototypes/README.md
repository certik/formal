# Stage-fusion prototypes

`stage_fusion_pair.cu` holds standalone CUDA kernels written the way a
compiler pass would generate them, used to decide whether LFortran should
fuse consecutive Runge-Kutta stages into one kernel. It covers the interior
only and compares, at 2048^2 for one pair of stages:

- two launches of a plain per-stage kernel (22.4 us per stage),
- fusion with shared-memory tiles and barriers (slower at every size),
- fusion by register streaming down column strips (39.3 us per pair, about
  12% faster than the compiler's current per-stage kernels at 2048^2 and
  slower at 1024^2),
- streaming a single stage down a column (slower).

Every variant reproduces the checksum of two separate stages bit for bit.
The measurements are recorded in LFortran's
`doc/src/gpu_optimization_passes.md` on the `perf/gpu-stencil-passes` branch.

Build: `nvcc -O3 -arch=native stage_fusion_pair.cu -o pair`
