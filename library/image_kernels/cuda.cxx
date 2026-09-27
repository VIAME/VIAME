/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include "cuda.h"
#include "cuda_internal.h"
#include "cuda_resize_plan.h"
#include "denoise_weights.h"
#include "gaussian_kernel.h"
#include <climits>
#include <limits>
#include <mutex>
#include <vector>

namespace viame {
namespace image_kernels {
namespace cuda {
static_assert(sizeof(std::size_t) >= 8,
              "CUDA image kernels require a 64-bit host");
namespace {
using detail::check;
struct device_scope {
  int previous;
  explicit device_scope(int device) {
    check(cudaGetDevice(&previous));
    if (previous != device)
      check(cudaSetDevice(device));
  }
  ~device_scope() { cudaSetDevice(previous); }
};
struct buffer {
  void *pointer = nullptr;
  std::size_t capacity = 0;
  buffer() = default;
  buffer(buffer const &) = delete;
  buffer &operator=(buffer const &) = delete;
  ~buffer() { cudaFree(pointer); }
  void reserve(std::size_t bytes) {
    if (bytes <= capacity)
      return;
    void *replacement = nullptr;
    check(cudaMalloc(&replacement, bytes));
    cudaFree(pointer);
    pointer = replacement;
    capacity = bytes;
  }
};
std::size_t element_size(pixel_type type) {
  switch (type) {
  case pixel_type::uint8:
    return 1;
  case pixel_type::float32:
    return sizeof(float);
  }
  throw std::invalid_argument("unsupported CUDA pixel type");
}
void validate_shape(int w, int h, int c) {
  // Bound both coordinate arithmetic and launch dimensions.
  if (w <= 0 || h <= 0 || w > INT_MAX / 4 || h > INT_MAX / 4 || c < 1 ||
      c > 4 || std::uint64_t(w) * h * c > INT_MAX / 8)
    throw std::invalid_argument("CUDA image must be nonempty, have 1--4 "
                                "channels and at most INT_MAX/8 elements");
}
} // namespace

struct image::storage {
  int w, h, c, device;
  pixel_type type;
  void *pointer = nullptr;
  storage(int width, int height, int channels, pixel_type t, int d)
      : w(width), h(height), c(channels), device(d), type(t) {
    validate_shape(w, h, c);
    check(cudaMalloc(&pointer, std::size_t(w) * h * c * element_size(type)));
  }
  ~storage() {
    // Destructors must also be safe during driver shutdown/error unwinding.
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess)
      return;
    if (cudaSetDevice(device) == cudaSuccess)
      cudaFree(pointer);
    cudaSetDevice(previous);
  }
};
int image::width() const noexcept { return data_ ? data_->w : 0; }
int image::height() const noexcept { return data_ ? data_->h : 0; }
int image::channels() const noexcept { return data_ ? data_->c : 0; }
pixel_type image::type() const {
  if (!data_)
    throw std::invalid_argument("empty CUDA image");
  return data_->type;
}
void *image::device_data() const {
  if (!data_)
    throw std::invalid_argument("empty CUDA image");
  return data_->pointer;
}
int image::device() const {
  if (!data_)
    throw std::invalid_argument("empty CUDA image");
  return data_->device;
}
std::size_t image::row_bytes() const noexcept {
  return data_ ? std::size_t(data_->w) * data_->c *
                     (data_->type == pixel_type::uint8 ? 1 : sizeof(float))
               : 0;
}

int device_count() noexcept {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess)
    return 0;
  return count;
}
std::string availability_error() {
  int count = 0;
  auto error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess)
    return cudaGetErrorString(error);
  return count ? std::string() : "No CUDA device available";
}

struct context::implementation {
  int device;
  cudaStream_t stream = nullptr;
  std::mutex mutex;
  // Destroy these while this context's device is selected.
  struct workspace {
    buffer gaussian_rows, taps, nlm_rows, nlm_sums, weights;
    buffer mean5, mean30, previous, resize_x, resize_y, resize_xoff,
        resize_yoff;
    int motion_width = 0, motion_height = 0, motion_channels = 0,
        motion_count = 0;
    int gaussian_size = 0, patch = 0, window = 0, channels = 0, shift = 0,
        levels = 0;
    double sigma = 0, strength = -1;
  };
  std::unique_ptr<workspace> scratch;
  explicit implementation(int d) : device(d) {
    if (d < 0)
      throw std::invalid_argument("CUDA device index must be nonnegative");
    device_scope scope(device);
    scratch = std::make_unique<workspace>();
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~implementation() {
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess &&
        cudaSetDevice(device) == cudaSuccess) {
      cudaStreamSynchronize(stream);
      scratch.reset();
      cudaStreamDestroy(stream);
      cudaSetDevice(previous);
    }
  }
  void validate(image const &im) const {
    if (im.device() != device)
      throw std::invalid_argument("image belongs to another CUDA device");
  }
  void compatible(image const &src, image const &dst) const {
    validate(dst);
    if (src.width() != dst.width() || src.height() != dst.height() ||
        src.channels() != dst.channels() || src.type() != dst.type())
      throw std::invalid_argument(
          "CUDA output must match the input shape and pixel type");
  }
  // Even if a launch fails after earlier work was queued, no call may return
  // while another context could observe incomplete writes to its images.
  struct completion {
    cudaStream_t stream;
    ~completion() { cudaStreamSynchronize(stream); }
    void wait() { check(cudaStreamSynchronize(stream)); }
  };
};
context::context(int device)
    : impl_(std::make_unique<implementation>(device)) {}
context::~context() = default;
image context::allocate(int width, int height, int channels, pixel_type type) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  device_scope scope(impl_->device);
  image result;
  result.data_ = std::make_shared<image::storage>(width, height, channels, type,
                                                  impl_->device);
  return result;
}
void context::upload(image const &dst, void const *host, std::size_t stride) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->validate(dst);
  if (!stride)
    stride = dst.row_bytes();
  if (!host || stride < dst.row_bytes() ||
      stride > std::numeric_limits<std::size_t>::max() / dst.height())
    throw std::invalid_argument("invalid CUDA upload buffer or stride");
  device_scope scope(impl_->device);
  implementation::completion done{impl_->stream};
  if (stride == dst.row_bytes())
    check(cudaMemcpyAsync(dst.data_->pointer, host, stride * dst.height(),
                          cudaMemcpyHostToDevice, impl_->stream));
  else
    check(cudaMemcpy2DAsync(dst.data_->pointer, dst.row_bytes(), host, stride,
                            dst.row_bytes(), dst.height(),
                            cudaMemcpyHostToDevice, impl_->stream));
  done.wait();
}
void context::download(image const &src, void *host, std::size_t stride) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->validate(src);
  if (!stride)
    stride = src.row_bytes();
  if (!host || stride < src.row_bytes() ||
      stride > std::numeric_limits<std::size_t>::max() / src.height())
    throw std::invalid_argument("invalid CUDA download buffer or stride");
  device_scope scope(impl_->device);
  implementation::completion done{impl_->stream};
  if (stride == src.row_bytes())
    check(cudaMemcpyAsync(host, src.data_->pointer, stride * src.height(),
                          cudaMemcpyDeviceToHost, impl_->stream));
  else
    check(cudaMemcpy2DAsync(host, stride, src.data_->pointer, src.row_bytes(),
                            src.row_bytes(), src.height(),
                            cudaMemcpyDeviceToHost, impl_->stream));
  done.wait();
}
image context::gaussian_blur(image const &src, int size, double sigma,
                             image const *output) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->validate(src);
  if (src.type() != pixel_type::float32 || size < 1 || size > 255 ||
      size % 2 != 1 || !std::isfinite(sigma) || sigma < 0)
    throw std::invalid_argument("CUDA Gaussian requires float32, odd size "
                                "1--255 and finite sigma >= 0");
  device_scope scope(impl_->device);
  implementation::completion done{impl_->stream};
  image dst;
  if (output) {
    impl_->compatible(src, *output);
    dst = *output;
  } else
    dst.data_ = std::make_shared<image::storage>(
        src.width(), src.height(), src.channels(), src.type(), impl_->device);
  auto &ws = *impl_->scratch;
  ws.gaussian_rows.reserve(src.row_bytes() * src.height());
  if (ws.gaussian_size != size || ws.sigma != sigma) {
    // A failed update must not reuse stale cached coefficients.
    ws.gaussian_size = 0;
    auto exact = gaussian_kernel_1d(size, sigma);
    std::vector<float> taps(exact.begin(), exact.end());
    ws.taps.reserve(taps.size() * sizeof(float));
    check(cudaMemcpyAsync(ws.taps.pointer, taps.data(),
                          taps.size() * sizeof(float), cudaMemcpyHostToDevice,
                          impl_->stream));
    done.wait(); // The temporary host vector must survive the copy.
    ws.gaussian_size = size;
    ws.sigma = sigma;
  }
  detail::gaussian(static_cast<float const *>(src.data_->pointer),
                   static_cast<float *>(dst.data_->pointer),
                   static_cast<float *>(ws.gaussian_rows.pointer), src.width(),
                   src.height(), src.channels(),
                   static_cast<float const *>(ws.taps.pointer), size,
                   impl_->stream);
  done.wait();
  return dst;
}
image context::denoise_non_local_means(image const &src, double strength,
                                       int patch, int window,
                                       image const *output) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->validate(src);
  if (src.type() != pixel_type::uint8 || src.channels() > 3 || patch < 1 ||
      patch > 63 || window < 1 || window > 63 || !std::isfinite(strength) ||
      strength < 0)
    throw std::invalid_argument("CUDA NLM requires uint8, 1--3 channels, "
                                "patch/window 1--63 and finite strength >= 0");
  patch = patch / 2 * 2 + 1;
  window = window / 2 * 2 + 1;
  device_scope scope(impl_->device);
  implementation::completion done{impl_->stream};
  image dst;
  if (output) {
    impl_->compatible(src, *output);
    dst = *output;
  } else
    dst.data_ = std::make_shared<image::storage>(
        src.width(), src.height(), src.channels(), src.type(), impl_->device);
  auto &ws = *impl_->scratch;
  auto pixels = std::size_t(src.width()) * src.height();
  ws.nlm_rows.reserve(std::size_t(src.width()) * (src.height() + patch - 1) *
                      sizeof(std::uint64_t));
  ws.nlm_sums.reserve(pixels * (src.channels() + 1) * sizeof(std::int64_t));
  if (ws.strength != strength || ws.patch != patch || ws.window != window ||
      ws.channels != src.channels()) {
    ws.strength = -1; // A failed update must never reuse stale cached weights.
    auto table = viame::image_kernels::detail::make_nlm_weights(
        strength, src.channels(), patch, window);
    ws.weights.reserve(table.values.size() * sizeof(std::int64_t));
    check(cudaMemcpyAsync(ws.weights.pointer, table.values.data(),
                          table.values.size() * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice, impl_->stream));
    done.wait();
    ws.strength = strength;
    ws.patch = patch;
    ws.window = window;
    ws.channels = src.channels();
    ws.shift = table.shift;
    ws.levels = static_cast<int>(table.values.size());
  }
  detail::nlm(static_cast<unsigned char const *>(src.data_->pointer),
              static_cast<unsigned char *>(dst.data_->pointer), src.width(),
              src.height(), src.channels(), patch, window,
              static_cast<std::int64_t const *>(ws.weights.pointer), ws.shift,
              ws.levels, static_cast<std::uint64_t *>(ws.nlm_rows.pointer),
              static_cast<std::int64_t *>(ws.nlm_sums.pointer), impl_->stream);
  done.wait();
  return dst;
}

image context::gfit_motion(image const &src, image const *output) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->validate(src);
  if (src.type() != pixel_type::uint8)
    throw std::invalid_argument("GFIT motion requires uint8");
  validate_shape(src.width(), src.height(), 3);
  device_scope scope(impl_->device);
  implementation::completion done{impl_->stream};
  image dst;
  if (output) {
    impl_->validate(*output);
    if (output->width() != src.width() || output->height() != src.height() ||
        output->channels() != 3 || output->type() != pixel_type::uint8)
      throw std::invalid_argument("GFIT motion output must be uint8 HxWx3");
    dst = *output;
  } else
    dst.data_ = std::make_shared<image::storage>(
        src.width(), src.height(), 3, pixel_type::uint8, impl_->device);
  auto &ws = *impl_->scratch;
  if (ws.motion_width != src.width() || ws.motion_height != src.height() ||
      ws.motion_channels != src.channels())
    ws.motion_count = 0;
  auto pixels = std::size_t(src.width()) * src.height();
  ws.mean5.reserve(pixels * sizeof(double));
  ws.mean30.reserve(pixels * sizeof(double));
  ws.previous.reserve(pixels);
  try {
    detail::gfit_motion(static_cast<unsigned char const *>(src.data_->pointer),
                        static_cast<unsigned char *>(dst.data_->pointer),
                        static_cast<double *>(ws.mean5.pointer),
                        static_cast<double *>(ws.mean30.pointer),
                        static_cast<unsigned char *>(ws.previous.pointer),
                        pixels, src.channels(), ws.motion_count, impl_->stream);
    done.wait();
  } catch (...) {
    ws.motion_count = 0;
    throw;
  }
  ws.motion_width = src.width();
  ws.motion_height = src.height();
  ws.motion_channels = src.channels();
  ws.motion_count = std::min(30, ws.motion_count + 1);
  return dst;
}
void context::reset_gfit_motion() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->scratch->motion_count = 0;
}
image context::resize_letterbox(image const &src, int width, int height) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->validate(src);
  if (src.type() != pixel_type::uint8)
    throw std::invalid_argument("letterbox requires uint8");
  validate_shape(width, height, src.channels());
  auto plan =
      detail::make_letterbox_plan(src.width(), src.height(), width, height);
  device_scope scope(impl_->device);
  implementation::completion done{impl_->stream};
  image dst;
  dst.data_ = std::make_shared<image::storage>(width, height, src.channels(),
                                               src.type(), impl_->device);
  auto &ws = *impl_->scratch;
  auto copy = [&](buffer &target, auto const &values) {
    auto bytes = values.size() * sizeof(values[0]);
    target.reserve(bytes);
    check(cudaMemcpyAsync(target.pointer, values.data(), bytes,
                          cudaMemcpyHostToDevice, impl_->stream));
  };
  copy(ws.resize_x, plan.x.entries);
  copy(ws.resize_y, plan.y.entries);
  copy(ws.resize_xoff, plan.x.offsets);
  copy(ws.resize_yoff, plan.y.offsets);
  detail::letterbox(
      static_cast<unsigned char const *>(src.data_->pointer),
      static_cast<unsigned char *>(dst.data_->pointer), src.width(),
      src.height(), src.channels(), width, height, plan.width, plan.height,
      plan.left, plan.top, plan.area,
      static_cast<int const *>(ws.resize_xoff.pointer),
      static_cast<int const *>(ws.resize_yoff.pointer),
      static_cast<detail::resize_entry const *>(ws.resize_x.pointer),
      static_cast<detail::resize_entry const *>(ws.resize_y.pointer),
      impl_->stream);
  done.wait();
  return dst;
}
} // namespace cuda
} // namespace image_kernels
} // namespace viame
