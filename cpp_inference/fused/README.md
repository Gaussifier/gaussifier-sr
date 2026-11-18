# Fused CUDA runtime (torch-free)

`gaussifier_fused_infer` runs the v0.12.0 Gaussifier inference pipeline with hand-written
CUDA kernels and nothing but the CUDA runtime: no LibTorch, TensorRT or cuDNN. It implements
the GroupNorm-free ("frozen-GN") release of the model, in which every GroupNorm is a
calibrated per-channel affine folded into its conv.

This is a copy of the [base model's runtime](https://github.com/Gaussifier/gaussifier) with
one addition, the super-resolution head (below). Build options, size modes, video mode, the
validation flags, the base runtime's tests and its measured results are documented in the
base repository's [runtime README](https://github.com/Gaussifier/gaussifier/blob/main/cpp_inference/fused/README.md).

## Super-resolution: back-projection heads

The SR correction head (`gaussifier_sr`) does not see `image - render`. At every K-loop state
it sees the back-projection residual `up(LR - down(render))`: `down` is the antialiased
bicubic resize to the LR size and `up` the bicubic resize (a = -0.75, align_corners=False)
back to the render's size, as `torch.nn.functional.interpolate` computes them. The input
image is the clamped bicubic x2 upsample of LR.

- **Blob.** GFCH v3 adds a `u32 residual_mode` to the header (1 = back-projection). Its
  first convs keep all 17 inputs: the fold of image-residual heads (v1 / v2, 14 inputs), which
  merges `image - rendered` into the image and rendered weights, does not apply. v1 / v2
  blobs load and run as before.
- **Residual** (`backprojection.cu`). Per state that feeds the head, `k_bp_down` evaluates
  the antialiased downsample on the LR grid (PyTorch's windows, renormalised border weights
  and FP32 evaluation order) minus the observation, and `k_bp_up` the bicubic upsample with
  clamped source indices. The result is unit 2 of the packed head input, an 8-byte plane
  `[H][W][4]` FP16 (r, g, b, 0) after units 0 and 1. Both kernels are bit-identical to
  PyTorch on every size tested (`tests/test_fused.py`).
- **Head.** Both first-conv kernels (`head_in_conv_kernel` for small images, the fused
  `head_enc0_kernel` for large ones) read unit 2 as a third input unit: one m16n8k8 MMA per
  tap and n8 tile on top of the m16n8k16 of units 0 and 1, with lanes `t >= 2` holding the
  zero half of k. The small kernel still fits two blocks per SM (48960 B of shared memory);
  the large one reads the residual weights through L1 so that it does too.
- **Exact or approximate residual.** With the observation (`PipelineOptions::bp_observation`
  and `observation_chw()`, a non-null `observation` in the C ABI, `--observation LR.npy` in the
  CLI) the residual is exactly `up(LR - down(render))`. Without it the pipeline uses
  `input - up(down(render))`, which differs where the bicubic upsample of LR leaves [0, 1]:
  -0.007 dB on average at x2 on the DIV2K crops below, up to -0.06 dB on one crop. The Python
  wrapper always passes LR.
- **Count on the device.** `set_count_source(rate_sum, factor)` (C ABI: `count_source`,
  `count_factor`) sets the count to `rint(factor x rate sum)` of another workspace's forward
  map (`run_rate_graph`, `gfused_rate_sum_device`). An x2 step (`4 x own(LR)`) therefore runs
  without a host round trip. A new count does not rebuild the workspace or recapture the
  graph: the count is written on the device before every launch.

Build (CUDA 12.8 or 13.x, CMake 3.18 or newer; 89 for Ada, 120 for RTX 50-series):

```bash
cmake -S cpp_inference/fused -B cpp_inference/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build cpp_inference/build --target gaussifier_fused gaussifier_fused_infer
```

The library lands in `cpp_inference/build/libgaussifier_fused.so.1`, where
`gaussifier_sr.fused` looks for it (or set `GAUSSIFIER_SR_FUSED_LIBRARY`). Pass
`-DCMAKE_CUDA_COMPILER=/path/to/nvcc` if CMake does not find CUDA.

Weights: `python tools/export_fused.py CHECKPOINT --out DIR` writes `head.gfch` (v3),
`fm.gffm` (the base release's `fm_fp16.gffm` when stage 1 is frozen) and `export.json`, read
by `FusedSR.from_export(DIR)`. `FusedSR.released()` downloads the released head blob
(`sr_head_fp16.gfch`) and the base release's `fm_fp16.gffm`.

C ABI additions ([`python_api.h`](python_api.h)), device pointers and stream-ordered:
`gfused_clone`, `gfused_set_max_points`, `gfused_residual_mode`, `gfused_rate_sum_device`,
`gfused_read_rate_sum`, `gfused_run_device`, `gfused_read_run_stats`, and the test hooks
`gfused_backprojection_residual` and `gfused_head_device`.

Validation on an RTX 5090 (FP16 blobs, seed 1234): the 20 DIV2K-valid 512 x 512 centre crops
with MATLAB-bicubic LR, Y-PSNR with `scale` border pixels removed, against the PyTorch
reference `gaussifier_sr.loop`:

| | PyTorch | fused | difference (largest per crop) |
|---|---|---|---|
| x2 | 35.374 dB | 35.374 dB | +0.001 dB (0.05) |
| x4 (two x2 steps) | 29.433 dB | 29.434 dB | +0.001 dB (0.03) |

Larger outputs, which run the fused enc0 kernel: x2 at 768 x 1024 (6 Urban100 crops) +0.003 dB,
at 1356 x 2040 (4 DIV2K frames) +0.005 dB.

Throughput (median of 20 distinct images after warm-up, CUDA tensors in and out, wall clock
with synchronisation; it includes the LR forward map, the bicubic input and the copies):

| | output | fused | PyTorch (eager) | speed-up |
|---|---|---|---|---|
| x2 | 512 x 512 | 3.59 ms | 23.9 ms | 6.7x |
| x2 | 768 x 1024 | 11.2 ms | 65.8 ms | 5.9x |
| x2 | 1356 x 2040 | 39.8 ms | 238 ms | 6.0x |
| x4 | 512 x 512 -> 2048 x 2048 | 72.6 ms | 461 ms | 6.3x |

Against an image-residual head of the same shape (nsys, GPU time per call), the
back-projection mode adds 0.17 ms at x2 / 512 x 512 (first conv +0.11 ms, residual kernels
0.09 ms; the rest of the loop varies with the predictions) and 4.3 ms on the x4 cascade to
2048 x 2048 (fused enc0 kernel +3.7 ms, residual kernels 0.8 ms).

## Architecture

One `FusedPipeline::run(K)` (`pipeline.cu`) enqueues every stage on one stream, out of one
device arena, with no host synchronisation, allocation or copies. `run_graph(K)` captures the
whole run as one CUDA graph per (H, W, K) and replays it for every image of that size. In
video mode (`PipelineOptions::video`) each run advances one frame, with one graph for the
first frame and one for every warm frame after it.

1. **Forward map** (`forward_map.cu`, `conv_fused.cuh`, `conv_block.cuh`, `epilogues.cuh`).
   The residual U-Net backbone (base 32, depth 4) runs on one implicit-GEMM conv kernel
   (`launch_conv2`: `mma.sync` FP16 with FP32 accumulation, or e4m3 FP8 per layer from a
   precision plan). Max-pooling and bilinear-upsample + concat (resized to the skip's
   extent at odd decoder levels) happen inside the input staging, each block's 1x1
   projection is an extra K segment of its second conv (in the up blocks only the skip half:
   a 1x1 conv commutes with the bilinear upsample, so the upsampled half runs as a 1x1 conv at
   low resolution whose up2x conv2 adds before its epilogue, `ConvParams::up_add`), the pooled
   skip copies are written by the producer's epilogue, and the density / log-cov /
   RGB heads run in the epilogue of the last conv. When a block's conv2 is FP8, its conv1
   stores the intermediate as e4m3 with conv2's activation scale (`F8StoreEpilogue`,
   `InMode::kPlainF8`): half the bytes, bit-identical to quantising an FP16 copy. From 1024 x 1024 pixels the stem and down0
   run as single-kernel residual blocks and the last block runs in 128-row bands for L2 reuse.
2. **Point count and sampler** (`pipeline_sampler.cu`). n = floor(sum rate), capped at 0.40
   points/pixel. The equal-mass Voronoi (EMV) sampler (error-diffusion oversample,
   mass-preserving merge rounds with an exact-count contract, Lloyd polish, cell-mass
   weights) keeps every count-dependent quantity on the device, so one launch sequence
   serves any count up to the planned capacity. Voronoi cells are exact nearest-seed
   assignments from a grid hash. `sampler_assign_check.cu` checks them against brute force.
3. **K-loop renderer** (`kloop_render.cu`). It is a raw-pointer port of gaussifier's native
   simple-sum renderer (it includes its device helpers from `third_party/raster`). The
   projection pass also writes the point-count map. The raster writes the clamped render, its
   squared error against the input, and the per-state half of the packed head input.
4. **Correction head** (`correction_head.cu`, `head_conv.cuh`, `head_fullres.cuh`). The head
   is a main U-Net (hidden 32) plus an aux U-Net (hidden 24), with every GroupNorm folded.
   The first conv of both nets runs as one persistent launch from the packed FP16 input
   (planar halo, the aux net at native width 24, partial edge tiles). Up to 512 x 1024
   pixels, the aux net runs on a forked stream and the full-resolution enc0.b and dec1.b + 1x1
   output layers use lean persistent kernels. Larger images fuse the first conv with enc0.b
   and use the generic conv core. The aux net runs at native width 24 unless a precision plan
   makes its enc1.a FP8. In that case it runs the copy padded to 32. The K-1 head calls
   update the points' log-cov, colour and position (xy step 2.56 / max(H, W)).
5. **CUDA graph**. Fixed K is a straight-line graph. Adaptive K (`--psnr-target` /
   `--min-gain`, off by default) puts the head-update + render step in a conditional WHILE
   node that the GPU decides after each state (CUDA 12.4+).

## Run

`gaussifier_sr.fused.FusedSR` is the super-resolution interface: it makes the bicubic input,
sets the count from the LR image and runs the x4 cascade. The binary runs single passes:

```bash
gaussifier_fused_infer --fm fm_fp16.gffm --head sr_head_fp16.gfch --K 7 input.npy render.npy \
    --observation lr.npy --n-points N
```

`input.npy` is the clamped bicubic x2 upsample of `lr.npy` (float32 `[3, H, W]` in [0, 1]) and
`lr.npy` the LR image (`[3, H/2, W/2]`), which makes the residual exact. `--n-points` sets the
count (4 x the model's own count on the LR image for the released head). Each image prints one
JSON line with the count, per-state PSNR against the input and timings.
