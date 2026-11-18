// Back-projection residual of the super-resolution correction loop.
//
// A back-projection head (GFCH v3, residual mode 1) sees, for every K-loop state, the render's
// disagreement with the observed low-resolution image instead of image - render:
//
//   residual = up(LR - down(render))
//   down = F.interpolate(render, size=(h, w), mode="bicubic", align_corners=False,
//                        antialias=True)
//   up   = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)
//
// Both resamplers follow PyTorch's CUDA kernels: the antialiased one with its filter
// (Keys cubic, a = -0.5) stretched by the scale, windows truncated at the border and their
// weights renormalised (upsample_gen2d_aa_out_frame, horizontal pass first), the plain one with
// a = -0.75 and clamped source indices (upsample_bicubic2d_out_frame), in FP32 with the same
// evaluation order.
//
// Because up is linear, residual = up(LR) - up(down(render)). Without the observation the
// pipeline uses its input image (the clamped bicubic upsample of LR) for up(LR):
//   residual ~ image - up(down(render)),
// which differs from the exact form only where the bicubic upsample of LR leaves [0, 1].

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gfused {

// Scales the kernels support: H / h and W / w in [1, 4] (the antialiased window has at most
// 4 * scale + 1 taps).
bool backprojection_supported(int H, int W, int h, int w);

// render: 3 x H x W planar FP32. observation: 3 x h x w planar FP32 (exact form), or nullptr to
// use base (3 x H x W, the approximate form). lr_scratch: 3 x h x w FP32 workspace.
// Outputs (either may be nullptr): packed_res, the head's input unit 2 ([H][W][4] FP16: r, g,
// b, 0); res32, the residual as 3 x H x W planar FP32. Graph capturable.
cudaError_t backprojection_residual(const float* render, const float* observation,
                                    const float* base, int H, int W, int h, int w,
                                    float* lr_scratch, __half* packed_res, float* res32,
                                    cudaStream_t stream);

}  // namespace gfused
