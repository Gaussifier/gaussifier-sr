// C ABI for libgaussifier_fused: a synchronous host-array interface (gfused_run, gfused_read,
// video), and a stream-ordered device-pointer interface for GPU-resident callers such as
// super-resolution (gfused_run_device and friends, below).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define GFUSED_NOEXCEPT noexcept
#else
#define GFUSED_NOEXCEPT
#endif

typedef struct GfusedSession GfusedSession;

const char* gfused_error(void) GFUSED_NOEXCEPT;
int gfused_create(const char* fm, const char* head, int device,
                  GfusedSession** output) GFUSED_NOEXCEPT;
void gfused_destroy(GfusedSession* session) GFUSED_NOEXCEPT;
int gfused_run(GfusedSession* session, const float* input, int height, int width,
               int states, int64_t seed, int count) GFUSED_NOEXCEPT;
int gfused_run_with_size_mode(GfusedSession* session, const float* input, int height, int width,
                              int states, int64_t seed, int count, int size_mode) GFUSED_NOEXCEPT;

// Read the latest result for an image or video session. Supply H*W floats for density,
// 3*H*W for render, 2*N for each xy array, N for weights, and 3*N for all other arrays.
int gfused_read(GfusedSession* session, float* density, float* render, float* initial_xy,
                float* initial_cov, float* initial_rgb, float* xy, float* cov, float* rgb,
                float* colour, float* weights) GFUSED_NOEXCEPT;

enum { GFUSED_MAX_STATES = 16 };

typedef struct GfusedVideoStats {
  uint64_t frame;
  int32_t n_requested, n_target, n_target_limited, survivors;
  int32_t kills_for_next, births_for_next, shortfall_for_next, k_used;
  double retained_fraction, rate_sum;
  double psnr[GFUSED_MAX_STATES];  // First k_used entries.
} GfusedVideoStats;

// Opens an ordered, fixed-size stream sharing the source model. The stream can outlive the
// source session. size_mode is 0 for pad, 1 for exact. count=0 selects adaptive count.
// Release with gfused_video_destroy. Calls return -1 on error (see gfused_error).
int gfused_video_create(GfusedSession* source, int height, int width, int states, int64_t seed,
                        int count, int size_mode, double count_ema, double warm_attribute_blend,
                        GfusedSession** output) GFUSED_NOEXCEPT;
// Runs a single CHW float32 frame and returns N. Read the result before pushing another frame.
// A run failure invalidates the stream. An argument validation failure leaves it usable.
int gfused_video_push(GfusedSession* session, const float* input, int height, int width,
                      GfusedVideoStats* stats) GFUSED_NOEXCEPT;
// Read N IDs and slots in the same row order as gfused_read.
int gfused_video_read_tracks(GfusedSession* session, int64_t* ids, int32_t* slots) GFUSED_NOEXCEPT;
void gfused_video_destroy(GfusedSession* session) GFUSED_NOEXCEPT;

// ---- Device-pointer interface.
// Pointers are device memory on the session's device; `stream` is a cudaStream_t (NULL: the
// legacy default stream). Calls enqueue their work on `stream` and return without waiting
// unless noted, so the caller keeps inputs alive and reads outputs in stream order. Graphs are
// captured on the session's own stream (first run of a size / state count) and replayed on
// `stream`. Calls return -1 on error (see gfused_error).

// A session on the weights of `source` (no file reads), with its own workspace and stream.
int gfused_clone(GfusedSession* source, GfusedSession** output) GFUSED_NOEXCEPT;
// The largest point count the session accepts (default 1000000); call before its first run.
int gfused_set_max_points(GfusedSession* session, int max_points) GFUSED_NOEXCEPT;
// The loaded correction head's residual input: 0 = image - render, 1 = back-projection
// up(LR - down(render)) (super-resolution heads).
int gfused_residual_mode(GfusedSession* session) GFUSED_NOEXCEPT;
// Forward map of a 3 x H x W float32 image: leaves the model's own point count before
// flooring (the sum of its rate) on the device, for gfused_run_device's count source and
// gfused_read_rate_sum.
int gfused_rate_sum_device(GfusedSession* session, const float* image, int height, int width,
                           int size_mode, void* stream) GFUSED_NOEXCEPT;
// Waits for `stream` and reads that rate sum.
int gfused_read_rate_sum(GfusedSession* session, void* stream, double* rate_sum) GFUSED_NOEXCEPT;
// One run with `states` rendered states of a 3 x H x W float32 image.
//   observation  back-projection heads: the 3 x (H/2) x (W/2) low-resolution image whose
//                bicubic upsample is `image` (exact residual), or NULL to use `image` itself
//                (image - up(down(render))); ignored by image-residual heads.
//   count        > 0: that many Gaussians; 0 with count_source: rint(count_factor x the rate
//                sum of count_source's last gfused_rate_sum_device); 0 without: the model's
//                own count. Above the own count the log-covariance is rescaled.
//   render       3 x H x W float32 output (the final render), or NULL.
int gfused_run_device(GfusedSession* session, const float* image, const float* observation,
                      int height, int width, int states, int64_t seed, int count,
                      GfusedSession* count_source, double count_factor, int size_mode,
                      void* stream, float* render) GFUSED_NOEXCEPT;
// Waits for `stream` and reads the last gfused_run_device's point count, the model's own count
// of its input and the rate sum; fails if the sampler broke its count contract.
int gfused_read_run_stats(GfusedSession* session, void* stream, int* count, int* own_count,
                          double* rate_sum) GFUSED_NOEXCEPT;

// ---- Test hooks (device pointers, synchronous).
// Back-projection residual of a 3 x H x W render against a 3 x h x w observation, or, with
// observation NULL, against `image` (3 x H x W): see backprojection.cuh.
int gfused_backprojection_residual(const float* render, const float* image,
                                   const float* observation, int height, int width,
                                   int lr_height, int lr_width, void* stream,
                                   float* residual) GFUSED_NOEXCEPT;
// The session's correction head alone: 17 x H x W float32 input in the training channel order
// (image 3, density, log_cov 3, rgb 3, rendered 3, residual 3, count; for image-residual heads
// the residual channels must be image - rendered), 8 x H x W float32 output. schedule: -1 by
// size, 0 small-image, 1 large-image kernels; packed: 1 through the packed head input (as the
// pipeline feeds it), 0 the FP32 gather path.
int gfused_head_device(GfusedSession* session, const float* input, int height, int width,
                       int schedule, int packed, void* stream, float* output) GFUSED_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef GFUSED_NOEXCEPT
