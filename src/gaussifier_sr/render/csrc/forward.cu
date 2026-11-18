// Forward pass: tile binning, then tile-parallel accumulation.
//
//  1. project_and_count: one thread per Gaussian projects it (centre, conic,
//     3-sigma radius) and finds the tiles of its bounding square. A Gaussian
//     touching more than LARGE_TILE_THRESHOLD tiles is "large" and goes to an
//     image-wide list; a small one increments the counters of its tiles.
//  2. A scan of the counters gives each tile's offset in the tile list.
//  3. scatter_to_tiles writes every small Gaussian into the lists of its
//     tiles. The list holds tile_list_capacity(N) entries; if the small
//     Gaussians need more, an overflow flag is raised.
//  4. rasterize: one 16x16 thread block per tile and one thread per pixel sum
//     colour * alpha over the tile's small Gaussians and over all large ones,
//     or over all Gaussians after an overflow.
//
// A small Gaussian is therefore rendered only inside its tile rectangle, which
// can clip its alpha >= 1/255 ellipse (3 to 3.33 sigma) at tile edges. Large
// Gaussians, and every Gaussian after an overflow, are limited only by the
// alpha cutoff. The order of Gaussians in a list depends on atomics, so the
// floating-point summation order is not deterministic.

#include "common.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/device/device_scan.cuh>

namespace {

__global__ void project_and_count(
    const int num_points,
    const float2 *__restrict__ means,
    const float2 *__restrict__ scales,
    const float *__restrict__ rotation,
    const dim3 img_size,
    const dim3 tile_bounds,
    float2 *__restrict__ projected_xy,
    float3 *__restrict__ projected_conic,
    float *__restrict__ projected_radius,
    uint8_t *__restrict__ projected_valid,
    int32_t *__restrict__ tile_counts,
    int32_t *__restrict__ large_count,
    int32_t *__restrict__ large_ids
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    float2 center;
    float3 conic;
    float radius;
    const bool ok =
        project_gaussian(means[idx], scales[idx], rotation[idx], img_size, center, conic, radius);
    if (ok) {
        projected_xy[idx] = center;
        projected_conic[idx] = conic;
        projected_radius[idx] = radius;
        projected_valid[idx] = 1;
    } else {
        projected_xy[idx] = {0.f, 0.f};
        projected_conic[idx] = {0.f, 0.f, 0.f};
        projected_radius[idx] = 0.f;
        projected_valid[idx] = 0;
    }
    if (!ok || radius <= 0.f) {
        return;
    }

    uint2 tile_min, tile_max;
    tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    if (tile_count(tile_min, tile_max) > LARGE_TILE_THRESHOLD) {
        large_ids[atomicAdd(large_count, 1)] = idx;
        return;
    }
    for (unsigned ty = tile_min.y; ty < tile_max.y; ++ty) {
        for (unsigned tx = tile_min.x; tx < tile_max.x; ++tx) {
            atomicAdd(&tile_counts[ty * tile_bounds.x + tx], 1);
        }
    }
}

__global__ void scatter_to_tiles(
    const int num_points,
    const float2 *__restrict__ projected_xy,
    const float *__restrict__ projected_radius,
    const uint8_t *__restrict__ projected_valid,
    const dim3 tile_bounds,
    const int32_t *__restrict__ tile_offsets,
    int32_t *__restrict__ tile_cursors,
    const int32_t capacity,
    int32_t *__restrict__ overflow_flag,
    int32_t *__restrict__ tile_ids
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points || !projected_valid[idx]) {
        return;
    }
    const float radius = projected_radius[idx];
    if (radius <= 0.f) {
        return;
    }

    uint2 tile_min, tile_max;
    tile_bbox(projected_xy[idx], radius, tile_bounds, tile_min, tile_max);
    if (tile_count(tile_min, tile_max) > LARGE_TILE_THRESHOLD) {
        return;
    }
    for (unsigned ty = tile_min.y; ty < tile_max.y; ++ty) {
        for (unsigned tx = tile_min.x; tx < tile_max.x; ++tx) {
            const int32_t tile = ty * tile_bounds.x + tx;
            const int32_t slot = atomicAdd(&tile_cursors[tile], 1) + tile_offsets[tile];
            if (slot < capacity) {
                tile_ids[slot] = idx;
            } else {
                atomicExch(overflow_flag, 1);
            }
        }
    }
}

struct GaussianBatch {
    bool valid[TILE_PIXELS];
    float2 xy[TILE_PIXELS];
    float3 conic[TILE_PIXELS];
    float color[TILE_PIXELS][MAX_CHANNELS];
};

// Adds Gaussians ids[begin, end) to this thread's pixel (the indices themselves
// when ids is null), staging them through shared memory TILE_PIXELS at a time.
// begin and end are uniform over the block.
__device__ __forceinline__ void accumulate_list(
    const int begin,
    const int end,
    const int32_t *__restrict__ ids,
    const bool inside,
    const float px,
    const float py,
    const unsigned channels,
    const float2 *__restrict__ projected_xy,
    const float3 *__restrict__ projected_conic,
    const uint8_t *__restrict__ projected_valid,
    const float *__restrict__ colors,
    GaussianBatch &batch,
    float *pix_out
) {
    const int rank = threadIdx.y * TILE_SIZE + threadIdx.x;
    for (int start = begin; start < end; start += TILE_PIXELS) {
        __syncthreads();  // the previous batch has been consumed
        const int idx = start + rank;
        if (idx < end) {
            const int32_t g = ids == nullptr ? idx : ids[idx];
            batch.valid[rank] = projected_valid[g] != 0;
            batch.xy[rank] = projected_xy[g];
            batch.conic[rank] = projected_conic[g];
            for (unsigned c = 0; c < channels; ++c) {
                batch.color[rank][c] = colors[g * channels + c];
            }
        }
        __syncthreads();
        if (!inside) {
            continue;
        }

        const int count = min(TILE_PIXELS, end - start);
        for (int t = 0; t < count; ++t) {
            if (!batch.valid[t]) {
                continue;
            }
            const float3 conic = batch.conic[t];
            const float2 xy = batch.xy[t];
            const float2 delta = {xy.x - px, xy.y - py};
            const float sigma =
                0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                conic.y * delta.x * delta.y;
            if (sigma < 0.f || isnan(sigma) || isinf(sigma)) {
                continue;
            }
            const float alpha = __expf(-sigma);
            if (alpha < kAlphaMin) {
                continue;
            }
            for (unsigned c = 0; c < channels; ++c) {
                pix_out[c] += batch.color[t][c] * alpha;
            }
        }
    }
}

__global__ void rasterize(
    const dim3 tile_bounds,
    const dim3 img_size,
    const unsigned channels,
    const int32_t *__restrict__ tile_ids,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ large_ids,
    const int32_t *__restrict__ large_count,
    const float2 *__restrict__ projected_xy,
    const float3 *__restrict__ projected_conic,
    const uint8_t *__restrict__ projected_valid,
    const float *__restrict__ colors,
    const int num_points,
    const int32_t capacity,
    const int32_t *__restrict__ overflow_flag,
    float *__restrict__ out_img  // (C, H, W)
) {
    const int tile = blockIdx.y * tile_bounds.x + blockIdx.x;
    const unsigned i = blockIdx.y * TILE_SIZE + threadIdx.y;
    const unsigned j = blockIdx.x * TILE_SIZE + threadIdx.x;
    const bool inside = i < img_size.y && j < img_size.x;
    const float px = (float)j;
    const float py = (float)i;

    __shared__ GaussianBatch batch;
    float pix_out[MAX_CHANNELS] = {0.f};

    const bool overflow = __ldg(overflow_flag) != 0;
    if (overflow) {
        // The tile lists are incomplete: evaluate every Gaussian here.
        accumulate_list(
            0, num_points, nullptr, inside, px, py, channels, projected_xy,
            projected_conic, projected_valid, colors, batch, pix_out
        );
    } else {
        accumulate_list(
            tile_offsets[tile], min(tile_offsets[tile + 1], capacity), tile_ids, inside,
            px, py, channels, projected_xy, projected_conic, projected_valid, colors,
            batch, pix_out
        );
        accumulate_list(
            0, __ldg(large_count), large_ids, inside, px, py, channels, projected_xy,
            projected_conic, projected_valid, colors, batch, pix_out
        );
    }

    if (inside) {
        const int hw = img_size.x * img_size.y;
        const int pix = i * img_size.x + j;
        for (unsigned c = 0; c < channels; ++c) {
            out_img[c * hw + pix] = pix_out[c];
        }
    }
}

}  // namespace

at::Tensor render_forward(
    const at::Tensor &means,
    const at::Tensor &scales,
    const at::Tensor &rotation,
    const at::Tensor &colors,
    const int64_t height,
    const int64_t width
) {
    check_gaussians(means, scales, rotation, colors);
    const int num_points = (int)means.size(0);
    const int channels = (int)colors.size(1);
    TORCH_CHECK(height > 0 && width > 0, "height and width must be positive");
    TORCH_CHECK(
        height * width * channels < (int64_t{1} << 31), "the image has too many elements"
    );
    if (num_points == 0) {
        return at::zeros({channels, height, width}, means.options());
    }

    const c10::cuda::CUDAGuard device_guard(means.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const dim3 img_size((unsigned)width, (unsigned)height, 1);
    const dim3 tile_bounds(
        (unsigned)((width + TILE_SIZE - 1) / TILE_SIZE),
        (unsigned)((height + TILE_SIZE - 1) / TILE_SIZE),
        1
    );
    const int num_tiles = (int)(tile_bounds.x * tile_bounds.y);
    const int64_t capacity = tile_list_capacity(num_points);

    const auto f32 = means.options();
    const auto i32 = means.options().dtype(at::kInt);
    at::Tensor projected_xy = at::empty({num_points, 2}, f32);
    at::Tensor projected_conic = at::empty({num_points, 3}, f32);
    at::Tensor projected_radius = at::empty({num_points}, f32);
    at::Tensor projected_valid = at::empty({num_points}, means.options().dtype(at::kByte));
    at::Tensor large_ids = at::empty({num_points}, i32);
    at::Tensor tile_ids = at::empty({capacity}, i32);
    // tile_offsets[0] = 0 and the scan fills the rest.
    at::Tensor tile_offsets = at::zeros({num_tiles + 1}, i32);
    // Zeroed counters: tile counts, tile cursors, large count, overflow flag.
    at::Tensor counters = at::zeros({2 * num_tiles + 2}, i32);
    int32_t *tile_counts = counters.data_ptr<int32_t>();
    int32_t *tile_cursors = tile_counts + num_tiles;
    int32_t *large_count = tile_cursors + num_tiles;
    int32_t *overflow_flag = large_count + 1;

    size_t scan_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scan_bytes, tile_counts, tile_offsets.data_ptr<int32_t>() + 1, num_tiles,
        stream
    ));
    at::Tensor scan_temp =
        at::empty({(int64_t)std::max<size_t>(scan_bytes, 1)}, means.options().dtype(at::kByte));

    const int blocks = (num_points + GAUSSIAN_BLOCK - 1) / GAUSSIAN_BLOCK;
    project_and_count<<<blocks, GAUSSIAN_BLOCK, 0, stream>>>(
        num_points,
        (const float2 *)means.data_ptr<float>(),
        (const float2 *)scales.data_ptr<float>(),
        rotation.data_ptr<float>(),
        img_size,
        tile_bounds,
        (float2 *)projected_xy.data_ptr<float>(),
        (float3 *)projected_conic.data_ptr<float>(),
        projected_radius.data_ptr<float>(),
        projected_valid.data_ptr<uint8_t>(),
        tile_counts,
        large_count,
        large_ids.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        scan_temp.data_ptr(), scan_bytes, tile_counts, tile_offsets.data_ptr<int32_t>() + 1,
        num_tiles, stream
    ));

    scatter_to_tiles<<<blocks, GAUSSIAN_BLOCK, 0, stream>>>(
        num_points,
        (const float2 *)projected_xy.data_ptr<float>(),
        projected_radius.data_ptr<float>(),
        projected_valid.data_ptr<uint8_t>(),
        tile_bounds,
        tile_offsets.data_ptr<int32_t>(),
        tile_cursors,
        (int32_t)capacity,
        overflow_flag,
        tile_ids.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor out_img = at::empty({channels, height, width}, f32);
    rasterize<<<tile_bounds, dim3(TILE_SIZE, TILE_SIZE, 1), 0, stream>>>(
        tile_bounds,
        img_size,
        (unsigned)channels,
        tile_ids.data_ptr<int32_t>(),
        tile_offsets.data_ptr<int32_t>(),
        large_ids.data_ptr<int32_t>(),
        large_count,
        (const float2 *)projected_xy.data_ptr<float>(),
        (const float3 *)projected_conic.data_ptr<float>(),
        projected_valid.data_ptr<uint8_t>(),
        colors.data_ptr<float>(),
        num_points,
        (int32_t)capacity,
        overflow_flag,
        out_img.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out_img;
}
