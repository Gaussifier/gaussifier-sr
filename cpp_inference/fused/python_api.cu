// C ABI for the Python binding: the synchronous host-array interface and the stream-ordered
// device-pointer interface (see python_api.h).
#include "python_api.h"

#include "backprojection.cuh"
#include "pipeline.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

struct GfusedSession {
  explicit GfusedSession(int device, std::unique_ptr<gfused::FusedPipeline> pipeline)
      : device(device), pipeline(std::move(pipeline)) {}
  ~GfusedSession() {
    if (stream) cudaStreamDestroy(stream);
  }

  int device;
  cudaStream_t stream = nullptr;
  std::unique_ptr<gfused::FusedPipeline> pipeline;
  int count = 0;
  bool device_run = false;  // the last run was gfused_run_device (count read on demand)
  // A nonzero state count identifies a video session.
  int video_states = 0;
  uint64_t frame = 0;
  bool video_failed = false;
};

namespace {

thread_local char last_error[1024]{};
// Arena preparation uses device synchronization, which cannot overlap graph capture.
std::mutex runtime_mutex;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct DeviceGuard {
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&previous));
    check(cudaSetDevice(device));
  }
  ~DeviceGuard() { cudaSetDevice(previous); }
  int previous;
};

gfused::SizeMode parse_size_mode(int mode) {
  if (mode != static_cast<int>(gfused::SizeMode::Pad) &&
      mode != static_cast<int>(gfused::SizeMode::Exact))
    throw std::invalid_argument("invalid fused size mode");
  return static_cast<gfused::SizeMode>(mode);
}

void validate_run(int height, int width, int states, int count) {
  if (!gfused::FusedPipeline::supports_size(height, width) ||
      states < 1 || states > GFUSED_MAX_STATES || count < 0 || count > 1000000)
    throw std::invalid_argument("invalid fused image size, state count or point count");
}

gfused::PipelineStats run_frame(GfusedSession& session, const float* input, int states) {
  auto& pipe = *session.pipeline;
  const auto stream = session.stream;
  const size_t bytes = 3 * static_cast<size_t>(pipe.H()) * pipe.W() * sizeof(float);
  check(cudaMemcpyAsync(pipe.input_chw(), input, bytes, cudaMemcpyHostToDevice, stream));
  const auto run = pipe.run_graph(states, stream);
  const auto sync = cudaStreamSynchronize(stream);
  check(run);
  check(sync);
  gfused::PipelineStats stats;
  check(pipe.read_stats(&stats, stream));
  if (stats.count_error || stats.n < 1 || stats.n > pipe.n_cap())
    throw std::runtime_error("fused sampler failed its point count contract");
  return stats;
}

template <typename T>
void download(T* dest, const void* src, size_t elements, cudaStream_t stream) {
  auto status = cudaMemcpyAsync(dest, src, elements * sizeof(T), cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) {
    cudaStreamSynchronize(stream);
    check(status);
  }
}

template <typename Fn>
int call(Fn fn) noexcept {
  try {
    std::lock_guard<std::mutex> lock(runtime_mutex);
    return fn();
  } catch (const std::exception& e) {
    std::snprintf(last_error, sizeof(last_error), "%s", e.what());
    return -1;
  }
}

}  // namespace

#define GFUSED_API extern "C" __attribute__((visibility("default")))

GFUSED_API const char* gfused_error() noexcept { return last_error; }

GFUSED_API int gfused_create(const char* fm, const char* head, int device,
                             GfusedSession** output) noexcept {
  return call([&] {
    if (!fm || !head || !output)
      throw std::invalid_argument("expected fused weight paths and output handle");
    DeviceGuard guard(device);
    auto session = std::make_unique<GfusedSession>(device, std::make_unique<gfused::FusedPipeline>());
    check(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));
    gfused::PipelineOptions options;
    options.forward_map_path = fm;
    options.head_path = head;
    options.keep_initial = true;
    std::string error;
    if (auto status = session->pipeline->load(options, &error); status != cudaSuccess) {
      if (!error.empty()) throw std::runtime_error(error);
      check(status);
    }
    *output = session.release();
    return 0;
  });
}

GFUSED_API void gfused_destroy(GfusedSession* session) noexcept {
  if (!session) return;
  call([&] {
    DeviceGuard guard(session->device);
    delete session;
    return 0;
  });
}

GFUSED_API int gfused_run_with_size_mode(GfusedSession* session, const float* input, int height,
                                         int width, int states, int64_t seed, int count,
                                         int size_mode) noexcept {
  return call([&] {
    if (!session || session->video_states || !input)
      throw std::invalid_argument("expected an image session and input");
    DeviceGuard guard(session->device);
    auto& pipe = *session->pipeline;
    session->count = 0;
    session->device_run = false;
    validate_run(height, width, states, count);
    pipe.set_size_mode(parse_size_mode(size_mode));
    pipe.set_sampling(seed, count);
    pipe.set_keep_initial(true);  // gfused_read returns the initial state too
    pipe.set_observation_mode(false);
    pipe.set_count_source(nullptr, 1.0);
    if (pipe.H() != height || pipe.W() != width || !pipe.n_cap())
      check(pipe.prepare(height, width));
    return session->count = run_frame(*session, input, states).n;
  });
}

GFUSED_API int gfused_run(GfusedSession* session, const float* input, int height, int width,
                          int states, int64_t seed, int count) noexcept {
  return gfused_run_with_size_mode(session, input, height, width, states, seed, count,
                                   static_cast<int>(gfused::SizeMode::Exact));
}

GFUSED_API int gfused_read(GfusedSession* session, float* density, float* render, float* initial_xy,
                           float* initial_cov, float* initial_rgb, float* xy, float* cov,
                           float* rgb, float* colour, float* weights) noexcept {
  return call([&] {
    if (!session) throw std::invalid_argument("expected a fused session");
    DeviceGuard guard(session->device);
    auto& pipe = *session->pipeline;
    const size_t n = session->count;
    if (!n) throw std::logic_error("no fused result available");
    const size_t pixels = static_cast<size_t>(pipe.H()) * pipe.W();
    const auto out = pipe.outputs();
    const auto stream = session->stream;
    download(density, out.density, pixels, stream);
    download(render, out.render, 3 * pixels, stream);
    download(initial_xy, out.initial_xy, 2 * n, stream);
    download(initial_cov, out.initial_log_cov, 3 * n, stream);
    download(initial_rgb, out.initial_rgb, 3 * n, stream);
    download(xy, out.xy, 2 * n, stream);
    download(cov, out.log_cov, 3 * n, stream);
    download(rgb, out.rgb, 3 * n, stream);
    download(colour, out.colour, 3 * n, stream);
    download(weights, out.weights, n, stream);
    check(cudaStreamSynchronize(stream));
    return 0;
  });
}

GFUSED_API int gfused_video_create(GfusedSession* source, int height, int width, int states,
                                   int64_t seed, int count, int size_mode, double count_ema,
                                   double warm_attribute_blend, GfusedSession** output) noexcept {
  return call([&] {
    if (!source || source->video_states || !output)
      throw std::invalid_argument("expected an image session and video output handle");
    validate_run(height, width, states, count);
    const auto mode = parse_size_mode(size_mode);
    if (!(count_ema >= 0 && count_ema <= 1 &&
          warm_attribute_blend >= 0 && warm_attribute_blend <= 1))
      throw std::invalid_argument("video count_ema and warm_attribute_blend must be in [0, 1]");
    DeviceGuard guard(source->device);
    auto session = std::make_unique<GfusedSession>(
        source->device, source->pipeline->make_video_workspace(count_ema, warm_attribute_blend));
    auto& pipe = *session->pipeline;
    pipe.set_size_mode(mode);
    pipe.set_sampling(seed, count);
    check(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));
    check(pipe.prepare(height, width));
    check(pipe.prepare_video_graphs(states, session->stream));
    session->video_states = states;
    *output = session.release();
    return 0;
  });
}

GFUSED_API int gfused_video_push(GfusedSession* session, const float* input, int height,
                                 int width, GfusedVideoStats* result) noexcept {
  return call([&] {
    if (!session || !session->video_states || !input || !result)
      throw std::invalid_argument("expected a video session, frame and stats output");
    auto& pipe = *session->pipeline;
    if (height != pipe.H() || width != pipe.W())
      throw std::invalid_argument("all video frames must have the session dimensions");
    if (session->video_failed) throw std::logic_error("fused video session failed");
    DeviceGuard guard(session->device);
    const int previous_count = session->count;
    session->count = 0;
    session->video_failed = true;
    const auto stats = run_frame(*session, input, session->video_states);
    const auto& tracks = stats.tracks;
    GfusedVideoStats frame{};
    frame.frame = session->frame;
    frame.n_requested = stats.n_requested;
    frame.n_target = tracks.target;
    frame.n_target_limited = tracks.limited_target;
    frame.survivors = tracks.survivors;
    frame.kills_for_next = tracks.killed;
    frame.births_for_next = tracks.born;
    frame.shortfall_for_next = tracks.shortfall;
    frame.k_used = stats.k_used;
    frame.retained_fraction = previous_count ? static_cast<double>(tracks.survivors) / previous_count
                                            : 0.0;
    frame.rate_sum = stats.rate_sum;
    double sq[GFUSED_MAX_STATES];
    download(sq, pipe.outputs().sq_err, frame.k_used, session->stream);
    check(cudaStreamSynchronize(session->stream));
    for (int k = 0; k < frame.k_used; ++k) {
      const double mse = sq[k] / (3.0 * height * width);
      frame.psnr[k] = -10.0 * std::log10(std::max(mse, 1e-10));
    }
    *result = frame;
    ++session->frame;
    session->video_failed = false;
    return session->count = stats.n;
  });
}

GFUSED_API int gfused_video_read_tracks(GfusedSession* session, int64_t* ids,
                                        int32_t* slots) noexcept {
  return call([&] {
    if (!session || !session->video_states || !ids || !slots)
      throw std::invalid_argument("expected a video session and track output buffers");
    if (!session->count) throw std::logic_error("no fused video result available");
    DeviceGuard guard(session->device);
    const auto out = session->pipeline->outputs();
    const auto stream = session->stream;
    static_assert(sizeof(*out.slots) == sizeof(*slots));
    download(ids, out.ids, session->count, stream);
    download(slots, out.slots, session->count, stream);
    check(cudaStreamSynchronize(stream));
    return 0;
  });
}

GFUSED_API void gfused_video_destroy(GfusedSession* session) noexcept {
  gfused_destroy(session);
}

// ---------------------------------------------------------------- device-pointer interface

namespace {

cudaStream_t as_stream(void* stream) { return static_cast<cudaStream_t>(stream); }

GfusedSession& image_session(GfusedSession* session) {
  if (!session || session->video_states)
    throw std::invalid_argument("expected an image session");
  return *session;
}

void validate_device_run(const gfused::FusedPipeline& pipe, int height, int width, int states,
                         int count) {
  if (!gfused::FusedPipeline::supports_size(height, width) || states < 1 ||
      states > GFUSED_MAX_STATES || count < 0 || count > pipe.max_points())
    throw std::invalid_argument("invalid fused image size, state count or point count");
}

void prepare_size(gfused::FusedPipeline& pipe, int height, int width) {
  if (pipe.H() != height || pipe.W() != width || !pipe.n_cap())
    check(pipe.prepare(height, width));
}

}  // namespace

GFUSED_API int gfused_clone(GfusedSession* source, GfusedSession** output) noexcept {
  return call([&] {
    auto& src = image_session(source);
    if (!output) throw std::invalid_argument("expected an output handle");
    DeviceGuard guard(src.device);
    auto session = std::make_unique<GfusedSession>(src.device, src.pipeline->make_workspace());
    check(cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking));
    *output = session.release();
    return 0;
  });
}

GFUSED_API int gfused_set_max_points(GfusedSession* session, int max_points) noexcept {
  return call([&] {
    auto& s = image_session(session);
    if (max_points < 1 || max_points > (1 << 24))
      throw std::invalid_argument("max_points must be in [1, 16777216]");
    DeviceGuard guard(s.device);
    s.pipeline->set_max_points(max_points);
    return 0;
  });
}

GFUSED_API int gfused_residual_mode(GfusedSession* session) noexcept {
  return call([&] { return image_session(session).pipeline->residual_mode(); });
}

GFUSED_API int gfused_rate_sum_device(GfusedSession* session, const float* image, int height,
                                      int width, int size_mode, void* stream) noexcept {
  return call([&] {
    auto& s = image_session(session);
    if (!image) throw std::invalid_argument("expected an input image");
    DeviceGuard guard(s.device);
    auto& pipe = *s.pipeline;
    validate_device_run(pipe, height, width, 1, 0);
    pipe.set_size_mode(parse_size_mode(size_mode));
    prepare_size(pipe, height, width);
    const auto st = as_stream(stream);
    const size_t bytes = 3 * static_cast<size_t>(height) * width * sizeof(float);
    check(cudaMemcpyAsync(pipe.input_chw(), image, bytes, cudaMemcpyDeviceToDevice, st));
    check(pipe.run_rate_graph(st, s.stream));
    return 0;
  });
}

GFUSED_API int gfused_read_rate_sum(GfusedSession* session, void* stream,
                                    double* rate_sum) noexcept {
  return call([&] {
    auto& s = image_session(session);
    if (!rate_sum) throw std::invalid_argument("expected an output pointer");
    DeviceGuard guard(s.device);
    const float* src = s.pipeline->rate_sum_device();
    if (!src) throw std::logic_error("no fused rate sum available");
    float value = 0.f;
    const auto st = as_stream(stream);
    check(cudaMemcpyAsync(&value, src, sizeof(float), cudaMemcpyDeviceToHost, st));
    check(cudaStreamSynchronize(st));
    *rate_sum = value;
    return 0;
  });
}

GFUSED_API int gfused_run_device(GfusedSession* session, const float* image,
                                 const float* observation, int height, int width, int states,
                                 int64_t seed, int count, GfusedSession* count_source,
                                 double count_factor, int size_mode, void* stream,
                                 float* render) noexcept {
  return call([&] {
    auto& s = image_session(session);
    if (!image) throw std::invalid_argument("expected an input image");
    DeviceGuard guard(s.device);
    auto& pipe = *s.pipeline;
    validate_device_run(pipe, height, width, states, count);
    const float* rate_src = nullptr;
    if (count == 0 && count_source) {
      auto& src = image_session(count_source);
      if (src.device != s.device) throw std::invalid_argument("count source on another device");
      rate_src = src.pipeline->rate_sum_device();
      if (!rate_src || !(count_factor > 0.0) || !std::isfinite(count_factor))
        throw std::invalid_argument("count source without a rate sum, or invalid count factor");
    }
    s.count = 0;
    s.device_run = false;
    pipe.set_size_mode(parse_size_mode(size_mode));
    pipe.set_sampling(seed, count);
    pipe.set_keep_initial(false);
    const bool bp = pipe.residual_mode() == gfused::kResidualBackProjection;
    pipe.set_observation_mode(bp && observation != nullptr);
    prepare_size(pipe, height, width);
    if (bp && states > 1 && !pipe.lr_h())
      throw std::invalid_argument("a back-projection head needs an even image size");
    pipe.set_count_source(rate_src, count_factor);
    check(pipe.capture_graph(states, s.stream));  // on the session's capturable stream
    const auto st = as_stream(stream);
    const size_t pixels = static_cast<size_t>(height) * width;
    check(cudaMemcpyAsync(pipe.input_chw(), image, 3 * pixels * sizeof(float),
                          cudaMemcpyDeviceToDevice, st));
    if (float* obs = pipe.observation_chw()) {
      const size_t lr = static_cast<size_t>(pipe.lr_h()) * pipe.lr_w();
      check(cudaMemcpyAsync(obs, observation, 3 * lr * sizeof(float), cudaMemcpyDeviceToDevice,
                            st));
    }
    check(pipe.run_graph(states, st));
    if (render)
      check(cudaMemcpyAsync(render, pipe.outputs().render, 3 * pixels * sizeof(float),
                            cudaMemcpyDeviceToDevice, st));
    s.device_run = true;
    return 0;
  });
}

GFUSED_API int gfused_read_run_stats(GfusedSession* session, void* stream, int* count,
                                     int* own_count, double* rate_sum) noexcept {
  return call([&] {
    auto& s = image_session(session);
    if (!s.device_run) throw std::logic_error("no fused device run to read");
    DeviceGuard guard(s.device);
    auto& pipe = *s.pipeline;
    gfused::PipelineStats stats;
    check(pipe.read_stats(&stats, as_stream(stream)));
    if (stats.count_error || stats.n < 1 || stats.n > pipe.n_cap())
      throw std::runtime_error("fused sampler failed its point count contract");
    if (count) *count = stats.n;
    if (own_count) *own_count = stats.n_auto;
    if (rate_sum) *rate_sum = stats.rate_sum;
    return 0;
  });
}

GFUSED_API int gfused_backprojection_residual(const float* render, const float* image,
                                              const float* observation, int height, int width,
                                              int lr_height, int lr_width, void* stream,
                                              float* residual) noexcept {
  return call([&] {
    if (!render || !residual || (!image && !observation) ||
        !gfused::backprojection_supported(height, width, lr_height, lr_width))
      throw std::invalid_argument("invalid back-projection arguments");
    float* scratch = nullptr;
    check(cudaMalloc(&scratch, 3 * static_cast<size_t>(lr_height) * lr_width * sizeof(float)));
    const auto st = as_stream(stream);
    cudaError_t e = gfused::backprojection_residual(render, observation, image, height, width,
                                                    lr_height, lr_width, scratch, nullptr,
                                                    residual, st);
    const cudaError_t sync = cudaStreamSynchronize(st);
    cudaFree(scratch);
    check(e);
    check(sync);
    return 0;
  });
}

GFUSED_API int gfused_head_device(GfusedSession* session, const float* input, int height,
                                  int width, int schedule, int packed, void* stream,
                                  float* output) noexcept {
  return call([&] {
    auto& s = image_session(session);
    if (!input || !output) throw std::invalid_argument("expected input and output buffers");
    DeviceGuard guard(s.device);
    check(s.pipeline->run_head_only(input, height, width, schedule, packed != 0, output,
                                    as_stream(stream)));
    return 0;
  });
}
