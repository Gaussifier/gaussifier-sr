// Torch-free end-to-end Gaussifier inference: forward map -> point count -> EMV sampler ->
// K-state correction loop, all enqueued on one stream from one device arena, with no host
// synchronisation, allocation or host<->device copies inside run(), so the whole run is
// captured as a single CUDA graph per (H, W, K) and replayed for every image of that size.
// Video captures separate graphs for the first frame and for subsequent frames.
//
//   FusedPipeline p;
//   p.load(opts, &err);                  // forward-map GFFM + correction-head GFCH blobs
//   p.prepare(H, W);                     // one arena for everything the pipeline owns
//   cudaMemcpyAsync(p.input_chw(), host, 3*H*W*4, H2D, stream);
//   p.run_graph(K, stream);              // capture once, then replay
//   ... read p.outputs() after synchronising; the point count is p.outputs().count[0].
//
// Semantics: gaussifier render_loop._correction_render_loss with the frozen-GN release config
// (n = floor(sum rate), capped at 0.40 points/pixel and 1e6; EMV placement with seed; hard
// points; per-point log-cov / rgb sampled from the forward-map maps; weights from Voronoi cell
// masses; K rendered states with K-1 head updates; xy step 2.56 / max(H, W); colour
// premultiplied by weight and clamped).
//
// Super-resolution heads (back-projection residual, backprojection.cuh): the input is the
// bicubic upsample of a low-resolution image LR of size (H / bp_factor, W / bp_factor), and
// every head update sees up(LR - down(render)). Fill observation_chw() with LR and set
// PipelineOptions::bp_observation for the exact residual; without it the pipeline uses
// input - up(down(render)). The point count can come from another pipeline's forward map on
// device (set_count_source: n = rint(factor * rate sum), e.g. 4 x the model's own count on LR
// for x2), so a whole super-resolution step runs without a host round trip.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "correction_head.cuh"
#include "forward_map.cuh"
#include "kloop_render.cuh"
#include "pipeline_sampler.cuh"

namespace gfused {

enum class SizeMode { Pad = 0, Exact = 1 };

struct PipelineOptions {
  std::string forward_map_path;  // GFFM blob from scripts/export_fused_weights.py
  std::string head_path;         // GFCH blob from scripts/export_fused_head.py
  int64_t seed = 12345;
  // 0 = adaptive floor(sum rate); > 0 = fixed count. Read on the device at every launch
  // (PipelineCtrl::n_override), so changing it keeps the workspace and its CUDA graphs.
  int n_override = 0;
  // With n_override above the model's own count, shift the log-cov diagonal so the splatted
  // mass is kept (production semantics). false = pin the count only (validation).
  bool count_rescale = true;
  double points_per_pixel_max = 0.40;
  int render_n_max = 1000000;
  int lloyd_iters = 3;
  int knn_k = 8;
  int merge_rounds = 2;
  float xy_step_pixels = 2.56f;
  int max_k = 16;
  // Adaptive K (both <= 0: fixed K, the default). The K passed to run()/run_graph() is the cap.
  // After each rendered state k (1-based) the loop stops when
  //   psnr_k >= psnr_target                          (psnr_target > 0), or
  //   k >= 2 and psnr_k - psnr_{k-1} < min_gain      (min_gain > 0).
  // In run_graph() the loop is a CUDA conditional WHILE node decided on the GPU; in run()
  // (eager) it is a host loop that reads the decision back after each state.
  float psnr_target = 0.f;
  float min_gain = 0.f;
  bool keep_initial = false;  // retain initial points and attributes for host consumers
  // Validation hook: skip the forward map; the caller fills external_maps() before each run.
  bool external_maps = false;
  // Debug: before every head call verify the packed head input (contract C1, written by the
  // pipeline and the renderer) against __float2half of the FP32 sources, counting mismatching
  // values in PipelineStats::io_mismatch. Not for timing.
  bool check_head_io = false;
  SizeMode size_mode = SizeMode::Exact;
  // Back-projection heads only: the observation is (H / bp_factor) x (W / bp_factor); with
  // bp_observation the caller fills observation_chw() before every run (exact residual).
  int bp_factor = 2;
  bool bp_observation = false;
  bool video = false;
  VideoTrackOptions video_tracks;
  // A small history contribution anchors the correction loop to the current feature maps.
  float warm_attribute_blend = 0.1f;
};

// Forward-map output buffers (device), for external_maps mode.
struct PipelineMaps {
  float* density;   // H x W
  float* log_cov;   // 3 x H x W
  float* rgb;       // 3 x H x W
  float* rate_sum;  // [1]
};

// Device pointers to the final state (valid after the stream completes).
struct PipelineOutputs {
  const int* count;        // [1] number of Gaussians N
  const float2* xy;        // (N, 2) in [0, 1]
  const float* log_cov;    // (N, 3) [a, b, c]
  const float* rgb;        // (N, 3) latent colour (un-premultiplied)
  const float* colour;     // (N, 3) clamp(rgb * weight, 0, 1): what the renderer consumes
  const float* weights;    // (N,)
  const float* render;     // 3 x H x W final render, clamped to [0, 1]
  const float* density;    // H x W raw learned density
  const float2* initial_xy;
  const float* initial_log_cov;
  const float* initial_rgb;
  const double* sq_err;    // [K] sum of squared error of each rendered state vs the input
  const int64_t* ids;      // [N] stream-local IDs (video only)
  const int* slots;        // [N] reusable, stable slots (video only)
};

// Host-side diagnostics (synchronous read).
struct PipelineStats {
  int n = 0, n_requested = 0, n_auto = 0, n_init = 0;
  float rate_sum = 0.f;
  int ed_tile_size = 0, matched = 0, forced_adj = 0, forced_nn = 0, count_error = 0;
  int k_used = 0;        // rendered states of the last run (== K in fixed mode)
  int io_mismatch = -1;  // check_head_io: mismatching values over the run (-1: not checked)
  VideoTrackStats tracks{};
};

// Device-resident pipeline scalars.
struct PipelineCtrl {
  float rate_sum;
  int n_auto;
  float diag_shift;
  float density_sum;
  int k_done;     // adaptive: states rendered so far
  int stop;       // adaptive: last decision
  double sq_cur;  // adaptive: squared error of the state being rendered
  int io_mismatch;  // check_head_io
  int n_override;   // PipelineOptions::n_override (or the count source), written per launch
};

class FusedPipeline {
 public:
  FusedPipeline() = default;
  ~FusedPipeline();
  FusedPipeline(const FusedPipeline&) = delete;
  FusedPipeline& operator=(const FusedPipeline&) = delete;

  cudaError_t load(const PipelineOptions& opts, std::string* error);
  // Creates an unprepared pipeline sharing the immutable model and options.
  std::unique_ptr<FusedPipeline> make_workspace() const;
  std::unique_ptr<FusedPipeline> make_video_workspace(double count_ema,
                                                      float warm_attribute_blend) const;
  // The neural nets use padded workspaces. All caller-visible buffers use the real dimensions.
  // Re-preparing releases the previous workspace and graphs.
  static bool supports_size(int H, int W);
  cudaError_t prepare(int H, int W, int max_points = 0);

  float* input_chw() { return in_chw_; }  // 3 x H x W float32 [0, 1], caller-filled
  // Back-projection heads: 3 x (H / bp_factor) x (W / bp_factor) float32 low-resolution
  // observation, caller-filled when PipelineOptions::bp_observation is set (else nullptr).
  float* observation_chw() { return opts_.bp_observation ? obs_ : nullptr; }
  int residual_mode() const;  // correction_head_residual_mode of the loaded head
  // Switches between the exact (observation) and approximate back-projection residual; keeps
  // the workspace, drops the captured graphs when it changes.
  void set_observation_mode(bool use_observation);
  // Forward map and rate sum only (the model's own count of the input before flooring), left
  // on the device at rate_sum_device(). Captured on `capture` (a capturable stream) on first
  // use, replayed on `stream`.
  cudaError_t run_rate_graph(cudaStream_t stream, cudaStream_t capture);
  const float* rate_sum_device() const { return ctrl_ ? &ctrl_->rate_sum : nullptr; }
  // Point count of the following runs from a device scalar: n = rint(factor * *rate_sum)
  // (then capped as usual, with the count-rescale above the own count). nullptr restores
  // PipelineOptions::n_override. Read at launch time; no recapture.
  void set_count_source(const float* rate_sum, double factor) {
    count_src_ = rate_sum;
    count_factor_ = factor;
  }
  // Workspace options; a change releases the prepared workspace and its graphs.
  void set_keep_initial(bool keep);
  void set_max_points(int max_points);  // PipelineOptions::render_n_max
  // Observation size of a back-projection head at the prepared size (0 if H or W is not a
  // multiple of bp_factor: the head cannot run, the forward map / rate sum still can).
  int lr_h() const { return lr_h_; }
  int lr_w() const { return lr_w_; }
  int max_points() const { return opts_.render_n_max; }
  // Test hook: the correction head alone on a 17 x H x W planar FP32 input in the training
  // channel order (image 3, density, log_cov 3, rgb 3, rendered 3, residual or image - rendered
  // 3, count) -> 8 x H x W planar FP32. packed: through the packed head input (as the pipeline
  // feeds it), else the FP32 gather path; schedule -1 by size, 0 small, 1 large. Synchronous,
  // with its own temporary workspace.
  cudaError_t run_head_only(const float* input17, int H, int W, int schedule, bool packed,
                            float* out8, cudaStream_t stream) const;
  PipelineMaps external_maps() {
    return {density_, external_cov_ ? external_cov_ : log_cov_, rgb_map_, &ctrl_->rate_sum};
  }
  // Enqueues one full run eagerly (also what the graph captures). In video mode each
  // call advances one frame. Keep the same stream across frames and prepare() to reset.
  cudaError_t run(int K, cudaStream_t stream);
  // Captures run(K) on first use for this K (in video mode: the graph of the next frame, first
  // or warm). run_graph also launches the graph.
  cudaError_t capture_graph(int K, cudaStream_t stream);
  cudaError_t run_graph(int K, cudaStream_t stream);
  // Changing sampling options invalidates the arena and its graphs, while keeping weights.
  void set_sampling(int64_t seed, int count);
  // Changing size mode invalidates a prepared odd-size arena and its graphs.
  void set_size_mode(SizeMode mode);
  // Capture both video graphs without advancing the stream, e.g. before benchmarking.
  cudaError_t prepare_video_graphs(int K, cudaStream_t stream);
  PipelineOutputs outputs() const;
  cudaError_t read_stats(PipelineStats* stats, cudaStream_t stream = nullptr) const;  // synchronous

  int H() const { return H_; }
  int W() const { return W_; }
  int n_cap() const { return n_cap_; }
  size_t arena_bytes() const { return arena_bytes_; }
  int graph_nodes(int K) const;

 private:
  void release();
  bool adaptive() const { return opts_.psnr_target > 0.f || opts_.min_gain > 0.f; }
  enum class NetSchedule { Aligned, ReflectPad, Exact };
  cudaError_t enqueue_forward_map(cudaStream_t s);
  cudaError_t run_frame(int K, bool warm, cudaStream_t s);
  cudaError_t capture_graph(int K, bool warm, cudaStream_t s);
  cudaError_t enqueue_initial_state(bool warm, bool pack, cudaStream_t s);
  cudaError_t enqueue_step(cudaStream_t s);              // head update + next render
  cudaError_t enqueue_decide(int K, cudaStream_t s, bool set_handle);
  cudaError_t run_adaptive(int K, bool warm, cudaStream_t s);
  cudaError_t enqueue_head(cudaStream_t s);              // head + correction apply
  cudaError_t enqueue_pixel_prep(bool warm, cudaStream_t s);  // shift, sampling density
  cudaError_t save_initial(cudaStream_t s);
  cudaError_t enqueue_pack(cudaStream_t s);              // static half of the packed head input
  void drop_graphs();
  bool bp() const { return residual_mode() == kResidualBackProjection; }
  int k_fixed_ = 0;
  unsigned long long cond_handle_ = 0;  // cudaGraphConditionalHandle while capturing
  void plan(Arena& a);

  PipelineOptions opts_;
  struct Model;
  std::shared_ptr<const Model> model_;
  ForwardMapWorkspace fm_ws_{};
  CorrectionHeadWorkspace head_ws_{};

  int H_ = 0, W_ = 0, net_h_ = 0, net_w_ = 0, n_cap_ = 0;
  NetSchedule schedule_ = NetSchedule::Aligned;
  uint8_t* arena_ = nullptr;
  size_t arena_bytes_ = 0;
  SamplerBuffers sb_{};
  VideoTracks tracks_{};
  float* attribute_history_ = nullptr;
  bool video_started_ = false;
  bool last_frame_warm_ = false;
  RenderBuffers rb_{};
  PipelineCtrl* ctrl_ = nullptr;
  float *in_chw_ = nullptr, *in_hwc_ = nullptr;
  float *density_ = nullptr, *log_cov_ = nullptr, *rgb_map_ = nullptr;
  float* external_cov_ = nullptr;
  float *net_image_ = nullptr, *net_cov_ = nullptr, *net_rgb_ = nullptr;
  float* real_logits_ = nullptr;
  float2* xy_ = nullptr;
  float *cov_ = nullptr, *rgb_ = nullptr, *weights_ = nullptr, *colour_ = nullptr;
  float *render_ = nullptr, *count_map_ = nullptr, *corr_ = nullptr;
  float2* initial_xy_ = nullptr;
  float *initial_cov_ = nullptr, *initial_rgb_ = nullptr;
  double* sq_err_ = nullptr;
  __half* packed16_ = nullptr;  // [2][H][W][8] packed head input (C1) (+ [H][W][4] unit 2)
  // back-projection heads: observation size, observation, LR-grid scratch, FP32 residual
  // (check_head_io only)
  int lr_h_ = 0, lr_w_ = 0;
  float *obs_ = nullptr, *bp_lr_ = nullptr, *bp_res32_ = nullptr;
  const float* count_src_ = nullptr;
  double count_factor_ = 1.0;
  std::map<std::pair<int, bool>, cudaGraphExec_t> graphs_;
  std::map<std::pair<int, bool>, size_t> graph_nodes_;
};

}  // namespace gfused
