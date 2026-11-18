// Host entry points of the 2D Gaussian simple-sum renderer.
#pragma once

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>

// Upper bound on colour channels (per-pixel accumulators in the kernels).
#define MAX_CHANNELS 12

// Renders N Gaussians to a float32 (C, H, W) image. means and scales are
// (N, 2), rotation holds N values, colors is (N, C) with 1 <= C <= 12; all
// float32, contiguous and on the same CUDA device.
at::Tensor render_forward(
    const at::Tensor &means,
    const at::Tensor &scales,
    const at::Tensor &rotation,
    const at::Tensor &colors,
    int64_t height,
    int64_t width
);

// Length of the tile list. When the small Gaussians' tile hits need more
// entries, every Gaussian is evaluated on every tile instead.
inline int64_t tile_list_capacity(const int64_t num_points) {
    return std::max<int64_t>(8 * num_points, 4096);
}

inline void check_gaussians(
    const at::Tensor &means,
    const at::Tensor &scales,
    const at::Tensor &rotation,
    const at::Tensor &colors
) {
    const auto check = [&](const at::Tensor &t, const char *name) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
        TORCH_CHECK(t.device() == means.device(), name, " must be on the device of means");
    };
    check(means, "means");
    check(scales, "scales");
    check(rotation, "rotation");
    check(colors, "colors");
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 2, "means must be (N, 2)");
    const int64_t n = means.size(0);
    // 32-bit indexing: tile lists hold up to 8 N entries, colours N * C.
    TORCH_CHECK(n < (int64_t{1} << 28), "at most 2^28 Gaussians are supported");
    TORCH_CHECK(
        scales.dim() == 2 && scales.size(0) == n && scales.size(1) == 2,
        "scales must be (N, 2)"
    );
    TORCH_CHECK(rotation.numel() == n, "rotation must hold N values");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == n, "colors must be (N, C)");
    TORCH_CHECK(
        colors.size(1) >= 1 && colors.size(1) <= MAX_CHANNELS,
        "colors must have 1 to ", MAX_CHANNELS, " channels, got ", colors.size(1)
    );
}
