// Constants and device helpers of the renderer kernels.
//
// Conventions (mirrored by render/reference.py):
//   * means are normalised [x, y]; the pixel-space centre is (x * W, y * H),
//     and the pixel in row i, column j is evaluated at the point (j, i);
//   * cov = (R S)(R S)^T with S = diag(scale_x, scale_y) in pixels and
//     R = [[cos t, sin t], [-sin t, cos t]], so scale_x is the standard
//     deviation along [cos t, -sin t];
//   * alpha = exp(-0.5 d^T cov^-1 d) with d = centre - pixel, skipped when
//     below 1/255; colours are summed without normalisation or transmittance.
#pragma once

#include "render.h"

#include <cstdint>
#include <cuda_runtime.h>

// Pixels per tile edge; one forward thread block renders one tile.
#define TILE_SIZE 16
#define TILE_PIXELS (TILE_SIZE * TILE_SIZE)
// A Gaussian whose bounding square touches more tiles than this is "large":
// it is not binned but evaluated on every tile.
#define LARGE_TILE_THRESHOLD 32
// Threads per block of the per-Gaussian kernels.
#define GAUSSIAN_BLOCK 256

constexpr float kAlphaMin = 1.f / 255.f;

// Pixel-space centre, conic (upper triangle of cov^-1) and the 3-sigma radius
// used for tile binning. Returns false for a singular covariance.
inline __device__ bool project_gaussian(
    const float2 mean,
    const float2 scale,
    const float rotation,
    const dim3 img_size,
    float2 &center,
    float3 &conic,
    float &radius
) {
    center = {mean.x * img_size.x, mean.y * img_size.y};

    const float c = cosf(rotation);
    const float s = sinf(rotation);
    // M = R S, cov = M M^T.
    const float m00 = c * scale.x;
    const float m01 = s * scale.y;
    const float m10 = -s * scale.x;
    const float m11 = c * scale.y;
    const float cov_xx = m00 * m00 + m01 * m01;
    const float cov_xy = m00 * m10 + m01 * m11;
    const float cov_yy = m10 * m10 + m11 * m11;

    const float det = cov_xx * cov_yy - cov_xy * cov_xy;
    if (det == 0.f) {
        return false;
    }
    const float inv_det = 1.f / det;
    conic = {cov_yy * inv_det, -cov_xy * inv_det, cov_xx * inv_det};

    // Largest eigenvalue with the discriminant floored at 0.1; radius is
    // three of its standard deviations, rounded up.
    const float b = 0.5f * (cov_xx + cov_yy);
    const float v1 = b + sqrtf(fmaxf(0.1f, b * b - det));
    const float v2 = b - sqrtf(fmaxf(0.1f, b * b - det));
    radius = ceilf(3.f * sqrtf(fmaxf(v1, v2)));
    return true;
}

// Truncates a tile coordinate toward zero and clamps it to [0, bound].
inline __device__ unsigned clamp_tile(const float t, const unsigned bound) {
    return min((unsigned)max(0, (int)t), bound);
}

// Tiles touched by the square of half-width `radius` around `center` (pixels),
// as the half-open range [tile_min, tile_max) clamped to the tile grid.
inline __device__ void tile_bbox(
    const float2 center,
    const float radius,
    const dim3 tile_bounds,
    uint2 &tile_min,
    uint2 &tile_max
) {
    const float cx = center.x / (float)TILE_SIZE;
    const float cy = center.y / (float)TILE_SIZE;
    const float r = radius / (float)TILE_SIZE;
    tile_min.x = clamp_tile(cx - r, tile_bounds.x);
    tile_max.x = clamp_tile(cx + r + 1, tile_bounds.x);
    tile_min.y = clamp_tile(cy - r, tile_bounds.y);
    tile_max.y = clamp_tile(cy + r + 1, tile_bounds.y);
}

inline __device__ int32_t tile_count(const uint2 tile_min, const uint2 tile_max) {
    return static_cast<int32_t>(tile_max.x - tile_min.x) *
           static_cast<int32_t>(tile_max.y - tile_min.y);
}
