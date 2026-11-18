// Back-projection residual (see backprojection.cuh).
//
// Two passes per K-loop state: k_bp_down evaluates the antialiased bicubic downsample of the
// render on the low-resolution grid (minus the observation, when one is given), k_bp_up the
// bicubic upsample of that map on the render's grid (subtracted from the input image in the
// approximate form) and writes the head's packed residual unit.

#include "backprojection.cuh"

#include "pipeline_common.cuh"

namespace gfused {
namespace {

// PyTorch's upsample_antialias::BicubicFilterFunctor (Keys cubic, a = -0.5).
__device__ __forceinline__ float aa_cubic(float x) {
  const float a = -0.5f;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// _compute_weights_span + _compute_weights: the input window [xmin, xmin + xsize) of output
// index i and its normalised weights (zero beyond xsize). The division is IEEE (the build uses
// fast math).
template <int kTaps>
__device__ __forceinline__ void aa_window(int i, int in_size, float scale, float support,
                                          float invscale, int& xmin, int& xsize,
                                          float (&w)[kTaps]) {
  const float center = scale * (i + 0.5f);
  xmin = max(static_cast<int>(center - support + 0.5f), 0);
  xsize = min(static_cast<int>(center + support + 0.5f), in_size) - xmin;
  const float xmin_m_center = xmin - center;
  float total = 0.0f;
#pragma unroll
  for (int j = 0; j < kTaps; ++j) {
    const float v = j < xsize ? aa_cubic((j + xmin_m_center + 0.5f) * invscale) : 0.0f;
    w[j] = v;
    total += v;
  }
  if (total != 0.0f) {
#pragma unroll
    for (int j = 0; j < kTaps; ++j) w[j] = j < xsize ? __fdiv_rn(w[j], total) : 0.0f;
  }
}

struct DownParams {
  const float* render;  // 3 x H x W
  const float* obs;     // 3 x h x w or nullptr
  float* out;           // 3 x h x w
  int H, W, h, w;
  float scale_h, scale_w, support_h, support_w, inv_h, inv_w;
};

// One low-resolution pixel per thread, all three channels: per window row the horizontal sum,
// then the vertical sum of those (upsample_gen2d_aa_out_frame's order).
template <int kTaps>
__global__ void __launch_bounds__(256) k_bp_down(DownParams p) {
  const int ox = blockIdx.x * blockDim.x + threadIdx.x;
  const int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= p.w || oy >= p.h) return;
  int xmin, xsize, ymin, ysize;
  float wx[kTaps], wy[kTaps];
  aa_window<kTaps>(ox, p.W, p.scale_w, p.support_w, p.inv_w, xmin, xsize, wx);
  aa_window<kTaps>(oy, p.H, p.scale_h, p.support_h, p.inv_h, ymin, ysize, wy);
  const size_t P = static_cast<size_t>(p.H) * p.W, Q = static_cast<size_t>(p.h) * p.w;
  const size_t o = static_cast<size_t>(oy) * p.w + ox;
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float* src = p.render + c * P + static_cast<size_t>(ymin) * p.W + xmin;
    float acc = 0.0f;
#pragma unroll
    for (int y = 0; y < kTaps; ++y) {
      if (y >= ysize) break;
      const float* row = src + static_cast<size_t>(y) * p.W;
      float hs = __ldg(row) * wx[0];
#pragma unroll
      for (int j = 1; j < kTaps; ++j)
        if (j < xsize) hs += __ldg(row + j) * wx[j];
      acc = y == 0 ? hs * wy[0] : acc + hs * wy[y];
    }
    p.out[c * Q + o] = p.obs ? __ldg(p.obs + c * Q + o) - acc : acc;
  }
}

// get_cubic_upsampling_coefficients (A = -0.75).
__device__ __forceinline__ void cubic_coeffs(float t, float (&c)[4]) {
  const float A = -0.75f;
  auto conv1 = [A](float x) { return ((A + 2) * x - (A + 3)) * x * x + 1; };
  auto conv2 = [A](float x) { return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A; };
  const float x1 = t, x2 = 1.0f - t;
  c[0] = conv2(x1 + 1.0f);
  c[1] = conv1(x1);
  c[2] = conv1(x2);
  c[3] = conv2(x2 + 1.0f);
}

struct UpParams {
  const float* lr;    // 3 x h x w (down(render), or observation - down(render))
  const float* base;  // 3 x H x W input image (approximate form) or nullptr
  __half* packed;     // [H][W][4] or nullptr
  float* res32;       // 3 x H x W or nullptr
  int H, W, h, w;
  float scale_h, scale_w;
};

__global__ void __launch_bounds__(256) k_bp_up(UpParams p) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.W || y >= p.H) return;
  // area_pixel_compute_source_index(scale, dst, align_corners=False, cubic=True)
  const float real_x = p.scale_w * (x + 0.5f) - 0.5f;
  const float real_y = p.scale_h * (y + 0.5f) - 0.5f;
  const int in_x = static_cast<int>(floorf(real_x)), in_y = static_cast<int>(floorf(real_y));
  float cx[4], cy[4];
  cubic_coeffs(real_x - in_x, cx);
  cubic_coeffs(real_y - in_y, cy);
  int xs[4], ys[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    xs[k] = min(max(in_x - 1 + k, 0), p.w - 1);
    ys[k] = min(max(in_y - 1 + k, 0), p.h - 1);
  }
  const size_t P = static_cast<size_t>(p.H) * p.W, Q = static_cast<size_t>(p.h) * p.w;
  const size_t pix = static_cast<size_t>(y) * p.W + x;
  float res[3];
#pragma unroll
  for (int c = 0; c < 3; ++c) {
    const float* src = p.lr + c * Q;
    float coef[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float* row = src + static_cast<size_t>(ys[k]) * p.w;
      coef[k] = __ldg(row + xs[0]) * cx[0] + __ldg(row + xs[1]) * cx[1] +
                __ldg(row + xs[2]) * cx[2] + __ldg(row + xs[3]) * cx[3];
    }
    const float v = coef[0] * cy[0] + coef[1] * cy[1] + coef[2] * cy[2] + coef[3] * cy[3];
    res[c] = p.base ? __ldg(p.base + c * P + pix) - v : v;
  }
  if (p.packed) {
    const __half2 rg = __floats2half2_rn(res[0], res[1]);
    const __half2 b0 = __floats2half2_rn(res[2], 0.0f);
    reinterpret_cast<uint2*>(p.packed)[pix] =
        make_uint2(*reinterpret_cast<const unsigned*>(&rg), *reinterpret_cast<const unsigned*>(&b0));
  }
  if (p.res32) {
#pragma unroll
    for (int c = 0; c < 3; ++c) p.res32[c * P + pix] = res[c];
  }
}

}  // namespace

bool backprojection_supported(int H, int W, int h, int w) {
  return h >= 1 && w >= 1 && h <= H && w <= W && H <= 4 * h && W <= 4 * w;
}

cudaError_t backprojection_residual(const float* render, const float* observation,
                                    const float* base, int H, int W, int h, int w,
                                    float* lr_scratch, __half* packed_res, float* res32,
                                    cudaStream_t s) {
  if (!render || !lr_scratch || (!observation && !base) || !backprojection_supported(H, W, h, w))
    return cudaErrorInvalidValue;
  // Host-side scales exactly as PyTorch derives them for size= (FP32 input / output); the
  // antialias support is 2 * scale (bicubic filter size 4) and invscale 1 / scale in double.
  DownParams d{render, observation, lr_scratch, H, W, h, w};
  d.scale_h = static_cast<float>(H) / h;
  d.scale_w = static_cast<float>(W) / w;
  d.support_h = static_cast<float>(4 * 0.5 * d.scale_h);
  d.support_w = static_cast<float>(4 * 0.5 * d.scale_w);
  d.inv_h = static_cast<float>(1.0 / d.scale_h);
  d.inv_w = static_cast<float>(1.0 / d.scale_w);
  const dim3 block(32, 8);
  const dim3 grid_lr((w + 31) / 32, (h + 7) / 8);
  if (d.scale_h <= 2.0f && d.scale_w <= 2.0f) {
    k_bp_down<9><<<grid_lr, block, 0, s>>>(d);
  } else {
    k_bp_down<17><<<grid_lr, block, 0, s>>>(d);
  }
  UpParams u{lr_scratch, observation ? nullptr : base, packed_res, res32, H, W, h, w};
  u.scale_h = static_cast<float>(h) / H;
  u.scale_w = static_cast<float>(w) / W;
  const dim3 grid_hr((W + 31) / 32, (H + 7) / 8);
  k_bp_up<<<grid_hr, block, 0, s>>>(u);
  return cudaGetLastError();
}

}  // namespace gfused
