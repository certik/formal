// Prototype: per-stage naive kernel vs fused stage pair (overlapped tiling in x,
// streaming down rows in y with shared-memory row rings). Interior only.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define C1 (1.0f/24.0f)
#define C2 (-9.0f/8.0f)
#define C3 (9.0f/8.0f)
#define C4 (-1.0f/24.0f)
#define I1 (-7.0f/112.0f)
#define I2 (63.0f/112.0f)
struct P { int nx, ny, ld; float D, idx, idy, hdt, dt; };
// x face f (between cells f and f+1) at row j: D*gx - vx*ix
__device__ __forceinline__ float fx(const float *q, const float *vx, int ldq, int ldv, int f, int j, const P &p) {
  const float *r = q + (size_t)j*ldq;
  float a=r[f-1], b=r[f], c=r[f+1], d=r[f+2];
  float g = p.idx*(C1*a + C2*b + C3*c + C4*d);
  float v = I1*a + I2*b + I2*c + I1*d;
  return p.D*g - vx[(size_t)j*ldv + f]*v;
}
__device__ __forceinline__ float fy(const float *q, const float *vy, int ldq, int ldv, int i, int f, const P &p) {
  float a=q[(size_t)(f-1)*ldq+i], b=q[(size_t)f*ldq+i], c=q[(size_t)(f+1)*ldq+i], d=q[(size_t)(f+2)*ldq+i];
  float g = p.idy*(C1*a + C2*b + C3*c + C4*d);
  float v = I1*a + I2*b + I2*c + I1*d;
  return p.D*g - vy[(size_t)f*ldv + i]*v;
}
__device__ __forceinline__ float rhs_at(const float *q, const float *vx, const float *vy, int ldq, int ldvx, int ldvy, int i, int j, const P &p) {
  float x = p.idx*(C1*fx(q,vx,ldq,ldvx,i-2,j,p) + C2*fx(q,vx,ldq,ldvx,i-1,j,p) + C3*fx(q,vx,ldq,ldvx,i,j,p) + C4*fx(q,vx,ldq,ldvx,i+1,j,p));
  float y = p.idy*(C1*fy(q,vy,ldq,ldvy,i,j-2,p) + C2*fy(q,vy,ldq,ldvy,i,j-1,p) + C3*fy(q,vy,ldq,ldvy,i,j,p) + C4*fy(q,vy,ldq,ldvy,i,j+1,p));
  return x + y;
}
// stage kind 0: acc=r, q=base+hdt*r ; kind 1: acc+=2r, q=base+hdt*r
__global__ void stage_k(const float *__restrict__ qin, float *__restrict__ qout, const float *__restrict__ vx, const float *__restrict__ vy,
                        float *__restrict__ acc, const float *__restrict__ base, P p, int kind, int i0, int i1, int j0, int j1) {
  int i = i0 + blockIdx.x*32 + (threadIdx.x & 31), j = j0 + blockIdx.y*8 + (threadIdx.x >> 5);
  if (i > i1 || j > j1) return;
  float r = rhs_at(qin, vx, vy, p.ld, p.ld - 1 + 0, p.ld, i, j, p);
  size_t k = (size_t)j*p.ld + i;
  if (kind == 0) acc[k] = r; else acc[k] += 2.0f*r;
  qout[k] = base[k] + p.hdt*r;
}
// Fused pair: stage kind0 then kind1. Block of NT threads covers NT columns of stage-1
// (NT-6 output columns) and marches rows [jb, je] of the output.
template <int NT, int RQ, int R1>
__global__ void pair_k(const float *__restrict__ qin, float *__restrict__ qout, const float *__restrict__ vx, const float *__restrict__ vy,
                       float *__restrict__ acc, const float *__restrict__ base, P p, int i0, int i1, int j0, int j1, int H) {
  __shared__ float sq[RQ][NT + 6];     // input rows ring
  __shared__ float s1[R1][NT];         // stage-1 output ring
  int W = NT - 6;
  int cbase = i0 + blockIdx.x*W - 3;   // stage-1 column of thread 0
  int t = threadIdx.x;
  int jb = j0 + blockIdx.y*H, je = min(jb + H - 1, j1);
  int c1 = cbase + t;                  // this thread's stage-1 column
  // stage-1 rows needed: jb-3 .. je+3 ; input rows: jb-6 .. je+6
  int ldq = p.ld, ldvx = p.ld - 1;
  for (int rr = jb - 6; rr <= je + 6 + 6; rr++) {
    // 1. load input row rr into ring (columns cbase-3 .. cbase+NT+2)
    if (rr <= je + 6) {
      for (int c = t; c < NT + 6; c += NT) sq[(rr + RQ*64) % RQ][c] = qin[(size_t)rr*ldq + cbase - 3 + c];
    }
    __syncthreads();
    // 2. stage 1 at row r1 = rr - 3 (needs input rows r1-3..r1+3)
    int r1 = rr - 3;
    if (r1 >= jb - 3 && r1 <= je + 3) {
      // x faces from ring row r1, y faces from ring column
      const float *row = sq[(r1 + RQ*64) % RQ];
      int lc = t + 3;   // local column in sq
      auto FX = [&](int lf) { // face between lf and lf+1 local
        float a=row[lf-1], b=row[lf], c=row[lf+1], d=row[lf+2];
        float g = p.idx*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vx[(size_t)r1*ldvx + cbase - 3 + lf]*v; };
      auto FY = [&](int f) {
        float a=sq[(f-1+RQ*64)%RQ][lc], b=sq[(f+RQ*64)%RQ][lc], c=sq[(f+1+RQ*64)%RQ][lc], d=sq[(f+2+RQ*64)%RQ][lc];
        float g = p.idy*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vy[(size_t)f*ldq + c1]*v; };
      float r = p.idx*(C1*FX(lc-2) + C2*FX(lc-1) + C3*FX(lc) + C4*FX(lc+1))
              + p.idy*(C1*FY(r1-2) + C2*FY(r1-1) + C3*FY(r1) + C4*FY(r1+1));
      size_t k = (size_t)r1*ldq + c1;
      float b = base[k];
      float q1 = b + p.hdt*r;
      s1[(r1 + R1*64) % R1][t] = q1;
      if (r1 >= jb && r1 <= je && t >= 3 && t < NT - 3 && c1 <= i1) acc[k] = r;   // kept for stage 2 below via reload
    }
    __syncthreads();
    // 3. stage 2 at row r2 = r1 - 3 (needs stage-1 rows r2-3..r2+3)
    int r2 = r1 - 3;
    if (r2 >= jb && r2 <= je && t >= 3 && t < NT - 3 && c1 <= i1) {
      auto Q = [&](int rowi, int lc) { return s1[(rowi + R1*64) % R1][lc]; };
      auto FX = [&](int lf) {
        float a=Q(r2,lf-1), b=Q(r2,lf), c=Q(r2,lf+1), d=Q(r2,lf+2);
        float g = p.idx*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vx[(size_t)r2*ldvx + cbase + lf]*v; };
      auto FY = [&](int f) {
        float a=Q(f-1,t), b=Q(f,t), c=Q(f+1,t), d=Q(f+2,t);
        float g = p.idy*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vy[(size_t)f*ldq + c1]*v; };
      float r = p.idx*(C1*FX(t-2) + C2*FX(t-1) + C3*FX(t) + C4*FX(t+1))
              + p.idy*(C1*FY(r2-2) + C2*FY(r2-1) + C3*FY(r2) + C4*FY(r2+1));
      size_t k = (size_t)r2*ldq + c1;
      acc[k] += 2.0f*r;
      qout[k] = base[k] + p.hdt*r;
    }
    __syncthreads();
  }
}

template <int TX, int TY>
__global__ void pair2_k(const float *__restrict__ qin, float *__restrict__ qout, const float *__restrict__ vx, const float *__restrict__ vy,
                       float *__restrict__ acc, const float *__restrict__ base, P p, int i0, int i1, int j0, int j1, int H) {
  constexpr int RG = 32;
  __shared__ float sq[RG][TX + 6];
  __shared__ float s1[RG][TX];
  int W = TX - 6;
  int cbase = i0 + blockIdx.x*W - 3;
  int tx = threadIdx.x % TX, ty = threadIdx.x / TX;
  int jb = j0 + blockIdx.y*H, je = min(jb + H - 1, j1);
  int c1 = cbase + tx;
  int ldq = p.ld, ldvx = p.ld - 1;
  for (int r0 = jb - 6; r0 <= je + 6; r0 += TY) {
    int rr = r0 + ty;
    if (rr <= je + 6) for (int c = tx; c < TX + 6; c += TX) sq[(rr + RG*64) % RG][c] = qin[(size_t)rr*ldq + cbase - 3 + c];
    __syncthreads();
    int r1 = rr - 3;
    if (r1 >= jb - 3 && r1 <= je + 3) {
      const float *row = sq[(r1 + RG*64) % RG];
      int lc = tx + 3;
      auto FX = [&](int lf) {
        float a=row[lf-1], b=row[lf], c=row[lf+1], d=row[lf+2];
        float g = p.idx*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vx[(size_t)r1*ldvx + cbase - 3 + lf]*v; };
      auto FY = [&](int f) {
        float a=sq[(f-1+RG*64)%RG][lc], b=sq[(f+RG*64)%RG][lc], c=sq[(f+1+RG*64)%RG][lc], d=sq[(f+2+RG*64)%RG][lc];
        float g = p.idy*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vy[(size_t)f*ldq + c1]*v; };
      float r = p.idx*(C1*FX(lc-2) + C2*FX(lc-1) + C3*FX(lc) + C4*FX(lc+1))
              + p.idy*(C1*FY(r1-2) + C2*FY(r1-1) + C3*FY(r1) + C4*FY(r1+1));
      size_t k = (size_t)r1*ldq + c1;
      s1[(r1 + RG*64) % RG][tx] = base[k] + p.hdt*r;
      if (r1 >= jb && r1 <= je && tx >= 3 && tx < TX - 3 && c1 <= i1) acc[k] = r;
    }
    __syncthreads();
    int r2 = r1 - 3;
    if (r2 >= jb && r2 <= je && tx >= 3 && tx < TX - 3 && c1 <= i1) {
      auto Q = [&](int rowi, int lc) { return s1[(rowi + RG*64) % RG][lc]; };
      auto FX = [&](int lf) {
        float a=Q(r2,lf-1), b=Q(r2,lf), c=Q(r2,lf+1), d=Q(r2,lf+2);
        float g = p.idx*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vx[(size_t)r2*ldvx + cbase + lf]*v; };
      auto FY = [&](int f) {
        float a=Q(f-1,tx), b=Q(f,tx), c=Q(f+1,tx), d=Q(f+2,tx);
        float g = p.idy*(C1*a + C2*b + C3*c + C4*d); float v = I1*a + I2*b + I2*c + I1*d;
        return p.D*g - vy[(size_t)f*ldq + c1]*v; };
      float r = p.idx*(C1*FX(tx-2) + C2*FX(tx-1) + C3*FX(tx) + C4*FX(tx+1))
              + p.idy*(C1*FY(r2-2) + C2*FY(r2-1) + C3*FY(r2) + C4*FY(r2+1));
      size_t k = (size_t)r2*ldq + c1;
      acc[k] += 2.0f*r;
      qout[k] = base[k] + p.hdt*r;
    }
    __syncthreads();
  }
}

// Register streaming: a warp owns 20 output columns (32 lanes, 6 halo lanes on each side),
// marches down H rows; y via rotating registers, x via shuffles; every face computed once.
#define SH(v, d) __shfl_sync(0xffffffffu, v, lane + (d))
__global__ void pairw_k(const float *__restrict__ qin, float *__restrict__ qout, const float *__restrict__ vx, const float *__restrict__ vy,
                       float *__restrict__ acc, const float *__restrict__ base, P p, int i0, int i1, int j0, int j1, int H) {
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int c = i0 - 6 + warp*20 + lane;
  int jb = j0 + blockIdx.y*H, je = min(jb + H - 1, j1);
  int ldq = p.ld, ldvx = p.ld - 1;
  bool col_ok = c <= i1 + 6;
  int cc = col_ok ? c : i1 + 6;
  float qa0, qa1, qa2, qa3;          // input rows r1..r1+3 (after shift)
  float fy0, fy1, fy2, fy3;          // stage-1 y faces r1-2..r1+1
  float pb0, pb1, pb2, pb3;          // stage-1 values rows r2..r2+3
  float gy0, gy1, gy2, gy3;          // stage-2 y faces r2-2..r2+1
  auto yface = [&](float a, float b, float cv, float d, int f, float vyv) {
    float g = p.idy*(C1*a + C2*b + C3*cv + C4*d); float v = I1*a + I2*b + I2*cv + I1*d;
    return p.D*g - vyv*v; };
  auto xface = [&](float a, float b, float cv, float d, float vxv) {
    float g = p.idx*(C1*a + C2*b + C3*cv + C4*d); float v = I1*a + I2*b + I2*cv + I1*d;
    return p.D*g - vxv*v; };
  // prologue: stage-1 row r1 = jb-3 needs input rows jb-6 .. jb ; y faces r1-2.. r1+1
  int r1 = jb - 3 - 3;   // we start the loop 3 rows earlier to fill the y faces of stage 1
  // fill input registers for rows (r1-1 .. r1+2) so that the first iteration loads r1+3
  qa0 = qin[(size_t)(r1-1)*ldq+cc]; qa1 = qin[(size_t)(r1)*ldq+cc]; qa2 = qin[(size_t)(r1+1)*ldq+cc]; qa3 = qin[(size_t)(r1+2)*ldq+cc];
  fy0 = fy1 = fy2 = fy3 = 0; pb0 = pb1 = pb2 = pb3 = 0; gy0 = gy1 = gy2 = gy3 = 0;
  for (; r1 <= je + 3; r1++) {
    // new input row r1+3 ; qa = rows r1, r1+1, r1+2, r1+3 ; keep row r1-? no longer needed
    float qn = qin[(size_t)(r1+3)*ldq+cc];
    float qm = qa0;  // row r1-1
    qa0 = qa1; qa1 = qa2; qa2 = qa3; qa3 = qn;   // rows r1..r1+3
    // y face r1+1 (rows r1..r1+3) ... needs row r1 .. r1+3 -> face f=r1+1 uses rows r1, r1+1, r1+2, r1+3
    float fnew = yface(qa0, qa1, qa2, qa3, r1+1, vy[(size_t)(r1+1)*ldq+cc]);
    fy0 = fy1; fy1 = fy2; fy2 = fy3; fy3 = fnew;    // faces r1-2..r1+1
    (void)qm;
    // x face at row r1: between c and c+1 needs cols c-1..c+2 of row r1 (qa0)
    float xm1 = SH(qa0, -1), xp1 = SH(qa0, 1), xp2 = SH(qa0, 2);
    float fxv = xface(xm1, qa0, xp1, xp2, vx[(size_t)r1*ldvx + cc]);
    float fx_m2 = SH(fxv, -2), fx_m1 = SH(fxv, -1), fx_p1 = SH(fxv, 1);
    float r = p.idx*(C1*fx_m2 + C2*fx_m1 + C3*fxv + C4*fx_p1) + p.idy*(C1*fy0 + C2*fy1 + C3*fy2 + C4*fy3);
    size_t k = (size_t)r1*ldq + cc;
    float bse = base[k];
    float q1 = bse + p.hdt*r;
    bool own1 = lane >= 6 && lane < 26 && col_ok;
    if (own1 && r1 >= jb && r1 <= je) acc[k] = r;
    // stage 2 at row r2 = r1 - 3: needs stage-1 rows r2..r2+3 for its new y face r2+1 -> rows r1-3..r1
    pb0 = pb1; pb1 = pb2; pb2 = pb3; pb3 = q1;      // stage-1 rows r1-3..r1
    int r2 = r1 - 3;
    float gnew = yface(pb0, pb1, pb2, pb3, r2+1, vy[(size_t)(r2+1)*ldq+cc]);
    gy0 = gy1; gy1 = gy2; gy2 = gy3; gy3 = gnew;    // faces r2-2..r2+1
    float ym1 = SH(pb0, -1), yp1 = SH(pb0, 1), yp2 = SH(pb0, 2);
    float gxv = xface(ym1, pb0, yp1, yp2, vx[(size_t)r2*ldvx + cc]);
    float g_m2 = SH(gxv, -2), g_m1 = SH(gxv, -1), g_p1 = SH(gxv, 1);
    float r2v = p.idx*(C1*g_m2 + C2*g_m1 + C3*gxv + C4*g_p1) + p.idy*(C1*gy0 + C2*gy1 + C3*gy2 + C4*gy3);
    if (own1 && r2 >= jb && r2 <= je) {
      size_t k2 = (size_t)r2*ldq + cc;
      acc[k2] += 2.0f*r2v;
      qout[k2] = base[k2] + p.hdt*r2v;
    }
  }
}

// Two columns per lane; NS = 1 (one stage) or 2 (fused pair). Useful lanes: [2*NS, 32-2*NS).
template <int NS>
__global__ void stream2_k(const float *__restrict__ qin, float *__restrict__ qout, const float *__restrict__ vx, const float *__restrict__ vy,
                       float *__restrict__ acc, const float *__restrict__ base, P p, int i0, int i1, int j0, int j1, int H, int kind1) {
  constexpr int HL = 2*NS;               // halo lanes each side
  constexpr int UW = 2*(32 - 2*HL);      // useful columns per warp
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int c = i0 - 2*HL + warp*UW + 2*lane;  // first of the two columns
  int jb = j0 + blockIdx.y*H, je = min(jb + H - 1, j1);
  int ldq = p.ld, ldvx = p.ld - 1;
  int cc = min(c, p.ld - 4);
  bool own = lane >= HL && lane < 32 - HL && c <= i1;
  bool own1 = own && c + 1 <= i1;
  auto yface = [&](float a, float b, float cv, float d, float vyv) {
    float g = p.idy*(C1*a + C2*b + C3*cv + C4*d); float v = I1*a + I2*b + I2*cv + I1*d;
    return p.D*g - vyv*v; };
  auto ld2 = [&](const float *a, size_t off) { return make_float2(a[off], a[off+1]); };
  // one stage on a row: input row value pair x (cols c, c+1), y faces (4 rows) -> rhs pair
  auto rhs2 = [&](float2 x, float2 f0, float2 f1, float2 f2, float2 f3, int row) {
    float l1 = __shfl_sync(~0u, x.y, lane - 1), r0 = __shfl_sync(~0u, x.x, lane + 1), r1 = __shfl_sync(~0u, x.y, lane + 1);
    float2 v = ld2(vx, (size_t)row*ldvx + cc);
    float fa = yface(l1, x.x, x.y, r0, v.x);      // face c   (xface same form with idx==idy here)
    float fb = yface(x.x, x.y, r0, r1, v.y);      // face c+1
    float lfa = __shfl_sync(~0u, fa, lane - 1), lfb = __shfl_sync(~0u, fb, lane - 1), rfa = __shfl_sync(~0u, fa, lane + 1);
    float2 o;
    o.x = p.idx*(C1*lfa + C2*lfb + C3*fa + C4*fb) + p.idy*(C1*f0.x + C2*f1.x + C3*f2.x + C4*f3.x);
    o.y = p.idx*(C1*lfb + C2*fa + C3*fb + C4*rfa) + p.idy*(C1*f0.y + C2*f1.y + C3*f2.y + C4*f3.y);
    return o; };
  auto yf2 = [&](float2 a, float2 b, float2 cv, float2 d, int f) {
    float2 v = ld2(vy, (size_t)f*ldq + cc);
    return make_float2(yface(a.x,b.x,cv.x,d.x,v.x), yface(a.y,b.y,cv.y,d.y,v.y)); };
  int r1 = jb - 3*(NS-1) - 3;
  float2 qa0 = ld2(qin,(size_t)(r1-1)*ldq+cc), qa1 = ld2(qin,(size_t)r1*ldq+cc), qa2 = ld2(qin,(size_t)(r1+1)*ldq+cc), qa3 = ld2(qin,(size_t)(r1+2)*ldq+cc);
  float2 z = make_float2(0,0), fy0=z, fy1=z, fy2=z, fy3=z, pb0=z, pb1=z, pb2=z, pb3=z, gy0=z, gy1=z, gy2=z, gy3=z;
  for (; r1 <= je + 3*(NS-1); r1++) {
    float2 qn = ld2(qin, (size_t)(r1+3)*ldq+cc);
    qa0 = qa1; qa1 = qa2; qa2 = qa3; qa3 = qn;
    fy0 = fy1; fy1 = fy2; fy2 = fy3; fy3 = yf2(qa0, qa1, qa2, qa3, r1+1);
    if (r1 < jb - 3*(NS-1)) continue;   // y faces warm-up
    float2 r = rhs2(qa0, fy0, fy1, fy2, fy3, r1);
    size_t k = (size_t)r1*ldq + cc;
    float2 b = ld2(base, k);
    if (NS == 1) {
      if (r1 >= jb && r1 <= je && own) {
        float2 a2 = kind1 ? ld2(acc,k) : z;
        acc[k] = kind1 ? a2.x + 2.0f*r.x : r.x; qout[k] = b.x + p.hdt*r.x;
        if (own1) { acc[k+1] = kind1 ? a2.y + 2.0f*r.y : r.y; qout[k+1] = b.y + p.hdt*r.y; }
      }
      continue;
    }
    float2 q1 = make_float2(b.x + p.hdt*r.x, b.y + p.hdt*r.y);
    if (own && r1 >= jb && r1 <= je) { acc[k] = r.x; if (own1) acc[k+1] = r.y; }
    pb0 = pb1; pb1 = pb2; pb2 = pb3; pb3 = q1;
    int r2 = r1 - 3;
    gy0 = gy1; gy1 = gy2; gy2 = gy3; gy3 = yf2(pb0, pb1, pb2, pb3, r2+1);
    if (r2 < jb) continue;
    float2 s = rhs2(pb0, gy0, gy1, gy2, gy3, r2);
    if (own && r2 <= je) {
      size_t k2 = (size_t)r2*ldq + cc;
      float2 b2 = ld2(base, k2);
      acc[k2] += 2.0f*s.x; qout[k2] = b2.x + p.hdt*s.x;
      if (own1) { acc[k2+1] += 2.0f*s.y; qout[k2+1] = b2.y + p.hdt*s.y; }
    }
  }
}

// warp-stream pair, RB rows per iteration (ILP): one column per lane, 20 useful lanes.
template <int RB>
__global__ void __launch_bounds__(128) pairc_k(const float *__restrict__ qin, float *__restrict__ qout, const float *__restrict__ vx, const float *__restrict__ vy,
                       float *__restrict__ acc, const float *__restrict__ base, P p, int i0, int i1, int j0, int j1, int H) {
  int lane = threadIdx.x & 31;
  int warp = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  int c = i0 - 6 + warp*20 + lane;
  int jb = j0 + blockIdx.y*H, je = min(jb + H - 1, j1);
  int ldq = p.ld, ldvx = p.ld - 1;
  int cc = min(c, p.ld - 3);
  bool own = lane >= 6 && lane < 26 && c <= i1;
  auto F = [&](float a, float b, float cv, float d, float vv, float id) {
    float g = id*(C1*a + C2*b + C3*cv + C4*d); float v = I1*a + I2*b + I2*cv + I1*d;
    return p.D*g - vv*v; };
  // window of input rows: qw[k] = row r0 - 3 + k, k = 0..RB+5 (stage-1 rows r0..r0+RB-1)
  // stage-1 y faces for rows r0..r0+RB-1 need faces r0-2..r0+RB: computed from qw.
  // stage-1 values window: sw[k] = row r0 - 6 + k (k=0..RB+5) for stage-2 rows r0-3..r0-3+RB-1
  float qw[RB + 6], sw[RB + 6];
  int r0 = jb - 3;                               // first stage-1 row
  #pragma unroll
  for (int k = 0; k < RB + 6; k++) sw[k] = 0.f;
  #pragma unroll
  for (int k = 0; k < 6; k++) qw[RB + k] = qin[(size_t)(r0 - 3 + k)*ldq + cc];   // rows r0-3..r0+2 at the end slots
  for (; r0 <= je + 3 + RB; r0 += RB) {
    // shift input window: keep last 6 rows (r0-3..r0+2), load RB new rows r0+3..r0+RB+2
    #pragma unroll
    for (int k = 0; k < 6; k++) qw[k] = qw[RB + k];
    #pragma unroll
    for (int k = 0; k < RB; k++) qw[6 + k] = qin[(size_t)(r0 + 3 + k)*ldq + cc];
    // stage-1 y faces f = r0-2 .. r0+RB : face f uses rows f-1..f+2 -> qw index f-1-(r0-3)
    float fy[RB + 3];
    #pragma unroll
    for (int k = 0; k < RB + 3; k++) { int f = r0 - 2 + k; fy[k] = F(qw[k], qw[k+1], qw[k+2], qw[k+3], vy[(size_t)f*ldq + cc], p.idy); }
    float q1[RB];
    #pragma unroll
    for (int k = 0; k < RB; k++) {
      int r = r0 + k; float x = qw[3 + k];
      float fx = F(__shfl_sync(~0u, x, lane - 1), x, __shfl_sync(~0u, x, lane + 1), __shfl_sync(~0u, x, lane + 2), vx[(size_t)r*ldvx + cc], p.idx);
      float rr = p.idx*(C1*__shfl_sync(~0u, fx, lane - 2) + C2*__shfl_sync(~0u, fx, lane - 1) + C3*fx + C4*__shfl_sync(~0u, fx, lane + 1))
               + p.idy*(C1*fy[k] + C2*fy[k+1] + C3*fy[k+2] + C4*fy[k+3]);
      size_t idx = (size_t)r*ldq + cc;
      q1[k] = base[idx] + p.hdt*rr;
      if (own && r >= jb && r <= je) acc[idx] = rr;
    }
    // stage-1 window: rows r0-6.. ; shift by RB, append q1
    #pragma unroll
    for (int k = 0; k < 6; k++) sw[k] = sw[RB + k];
    #pragma unroll
    for (int k = 0; k < RB; k++) sw[6 + k] = q1[k];
    // stage-2 rows r2 = r0-3 .. r0-3+RB-1 ; sw[k] = row r0-6+k ; y faces f = r0-5 .. r0-3+RB
    float gy[RB + 3];
    #pragma unroll
    for (int k = 0; k < RB + 3; k++) { int f = r0 - 5 + k; gy[k] = F(sw[k], sw[k+1], sw[k+2], sw[k+3], vy[(size_t)max(f,0)*ldq + cc], p.idy); }
    #pragma unroll
    for (int k = 0; k < RB; k++) {
      int r = r0 - 3 + k; float x = sw[3 + k];
      float fx = F(__shfl_sync(~0u, x, lane - 1), x, __shfl_sync(~0u, x, lane + 1), __shfl_sync(~0u, x, lane + 2), vx[(size_t)max(r,0)*ldvx + cc], p.idx);
      float rr = p.idx*(C1*__shfl_sync(~0u, fx, lane - 2) + C2*__shfl_sync(~0u, fx, lane - 1) + C3*fx + C4*__shfl_sync(~0u, fx, lane + 1))
               + p.idy*(C1*gy[k] + C2*gy[k+1] + C3*gy[k+2] + C4*gy[k+3]);
      if (own && r >= jb && r <= je) {
        size_t idx = (size_t)r*ldq + cc;
        acc[idx] += 2.0f*rr; qout[idx] = base[idx] + p.hdt*rr;
      }
    }
  }
}
#ifndef TXV
#define TXV 64
#endif
#ifndef TYV
#define TYV 8
#endif
int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2048, H = argc > 2 ? atoi(argv[2]) : 64;
  P p; p.nx = n; p.ny = n; p.ld = n + 2; p.D = 0.5f; p.idx = p.idy = n/6.28f; p.dt = 1e-7f; p.hdt = p.dt/2;
  size_t ns = (size_t)(n+2)*(n+2);
  float *h = (float*)malloc(ns*4);
  float *q0,*q1,*q2,*vx,*vy,*acc,*base,*accb;
  cudaMalloc(&q0,ns*4);cudaMalloc(&q1,ns*4);cudaMalloc(&q2,ns*4);cudaMalloc(&vx,ns*4);cudaMalloc(&vy,ns*4);cudaMalloc(&acc,ns*4);cudaMalloc(&accb,ns*4);cudaMalloc(&base,ns*4);
  for (size_t k=0;k<ns;k++) h[k]=1.0f+0.5f*sinf(0.013f*(k%(n+2)))*cosf(0.007f*(k/(n+2)));
  cudaMemcpy(q0,h,ns*4,cudaMemcpyHostToDevice); cudaMemcpy(base,h,ns*4,cudaMemcpyHostToDevice);
  for (size_t k=0;k<ns;k++) h[k]=0.3f*sinf(0.001f*k);
  cudaMemcpy(vx,h,ns*4,cudaMemcpyHostToDevice); cudaMemcpy(vy,h,ns*4,cudaMemcpyHostToDevice);
  cudaMemset(q1,0,ns*4); cudaMemset(q2,0,ns*4);
  int i0 = 8, i1 = n - 6, j0 = 8, j1 = n - 6;   // output region (interior only)
  // stage-1 of the reference must cover rows/cols j0-3..j1+3
  dim3 g1((i1+3 - (i0-3) + 1 + 31)/32, (j1+3-(j0-3)+1+7)/8), g2((i1-i0+1+31)/32,(j1-j0+1+7)/8);
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float ms;
  // reference: two per-stage launches
  for (int it = 0; it < 2; it++) {
    cudaEventRecord(e0);
    for (int r = 0; r < 100; r++) {
      stage_k<<<g1,256>>>(q0,q1,vx,vy,acc,base,p,0,i0-3,i1+3,j0-3,j1+3);
      stage_k<<<g2,256>>>(q1,q2,vx,vy,acc,base,p,1,i0,i1,j0,j1);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
  }
  printf("per-stage pair: %.2f us\n", ms*1000/100);
  cudaMemcpy(h,q2,ns*4,cudaMemcpyDeviceToHost);
  double s1=0; for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) s1 += h[(size_t)j*(n+2)+i]*(1+(i*7+j)%5);
  cudaMemset(q2,0,ns*4);
  dim3 gp((i1-i0+1 + TXV-6-1)/(TXV-6), (j1-j0+1+H-1)/H);
  for (int it = 0; it < 2; it++) {
    cudaEventRecord(e0);
    for (int r = 0; r < 100; r++) pair2_k<TXV,TYV><<<gp,TXV*TYV>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H);
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
  }
  cudaError_t err = cudaGetLastError(); if (err) printf("err %s\n", cudaGetErrorString(err));
  {
    int warps = (i1 - i0 + 1 + 19)/20; dim3 gw((warps + 3)/4, (j1-j0+1+H-1)/H);
    cudaMemset(q2,0,ns*4);
    for (int it = 0; it < 2; it++) {
      cudaEventRecord(e0);
      for (int r = 0; r < 100; r++) pairw_k<<<gw,128>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H);
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
    }
    printf("warp-stream pair H=%d: %.2f us\n", H, ms*1000/100);
    cudaMemcpy(h,q2,ns*4,cudaMemcpyDeviceToHost);
    double s3=0; for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) s3 += h[(size_t)j*(n+2)+i]*(1+(i*7+j)%5);
    printf("checksum warp %.10e\n", s3);
    for (int rb = 2; rb <= 8; rb *= 2) {
      cudaMemset(q2,0,(size_t)ns*4);
      for (int it = 0; it < 2; it++) {
        cudaEventRecord(e0);
        for (int r = 0; r < 100; r++) {
          if (rb == 2) pairc_k<2><<<gw,128>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H);
          if (rb == 4) pairc_k<4><<<gw,128>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H);
          if (rb == 8) pairc_k<8><<<gw,128>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H);
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
      }
      cudaMemcpy(h,q2,ns*4,cudaMemcpyDeviceToHost);
      double s5=0; for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) s5 += h[(size_t)j*(n+2)+i]*(1+(i*7+j)%5);
      printf("chunked pair RB=%d H=%d: %.2f us checksum %.10e\n", rb, H, ms*1000/100, s5);
    }
    for (int ns = 1; ns <= 2; ns++) {
      int UW = 2*(32 - 4*ns); int warps2 = (i1 - i0 + 1 + UW - 1)/UW; dim3 g2w((warps2 + 3)/4, (j1-j0+1+H-1)/H);
      cudaMemset(q2,0,ns*4);
      for (int it = 0; it < 2; it++) {
        cudaEventRecord(e0);
        for (int r = 0; r < 100; r++) {
          if (ns == 2) stream2_k<2><<<g2w,128>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H,0);
          else {
            int UW1 = 2*(32-4); int w1 = (i1+3 - (i0-3) + 1 + UW1 - 1)/UW1; dim3 g1w((w1+3)/4, (j1+3-(j0-3)+1+H-1)/H);
            stream2_k<1><<<g1w,128>>>(q0,q1,vx,vy,acc,base,p,i0-3,i1+3,j0-3,j1+3,H,0);
            int w2 = (i1 - i0 + 1 + UW1 - 1)/UW1; dim3 g2b((w2+3)/4, (j1-j0+1+H-1)/H);
            stream2_k<1><<<g2b,128>>>(q1,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H,1);
          }
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
      }
      cudaMemcpy(h,q2,ns*4,cudaMemcpyDeviceToHost);
      double s4=0; for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) s4 += h[(size_t)j*(n+2)+i]*(1+(i*7+j)%5);
      printf("stream2 %s H=%d: %.2f us checksum %.10e\n", ns == 2 ? "fused pair" : "two stages", H, ms*1000/100, s4);
    }
    cudaMemset(q2,0,ns*4);
    for (int it = 0; it < 2; it++) {
      cudaEventRecord(e0);
      for (int r = 0; r < 100; r++) pair2_k<TXV,TYV><<<gp,TXV*TYV>>>(q0,q2,vx,vy,acc,base,p,i0,i1,j0,j1,H);
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1);
    }
  }
  printf("fused pair2 TX=%d TY=%d H=%d", TXV, TYV, H); printf(": %.2f us\n", H, ms*1000/100);
  cudaMemcpy(h,q2,ns*4,cudaMemcpyDeviceToHost);
  double s2=0; for (int j=j0;j<=j1;j++) for (int i=i0;i<=i1;i++) s2 += h[(size_t)j*(n+2)+i]*(1+(i*7+j)%5);
  printf("checksum %.10e %.10e\n", s1, s2);
}
