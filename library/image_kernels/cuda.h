/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#ifndef VIAME_IMAGE_KERNELS_CUDA_H
#define VIAME_IMAGE_KERNELS_CUDA_H

#include "viame_image_kernels_cuda_export.h"
#include <cstddef>
#include <memory>
#include <string>

namespace viame {
namespace image_kernels {
namespace cuda {

enum class pixel_type { uint8, float32 };

/// Returns zero when the runtime or driver cannot provide a CUDA device.
VIAME_IMAGE_KERNELS_CUDA_EXPORT int device_count() noexcept;
/// Empty on success; otherwise the CUDA runtime's diagnostic.
VIAME_IMAGE_KERNELS_CUDA_EXPORT std::string availability_error();

/// An owning, interleaved device image. Copies share storage; no host pixels.
/// Images may outlive their context. Destruction releases the device
/// allocation.
class VIAME_IMAGE_KERNELS_CUDA_EXPORT image {
public:
  image() = default;
  int width() const noexcept;
  int height() const noexcept;
  int channels() const noexcept;
  pixel_type type() const;
  int device() const;
  std::size_t row_bytes() const noexcept;

private:
  struct storage;
  std::shared_ptr<storage> data_;
  friend class context;
};

/// Owns a nonblocking CUDA stream and reusable scratch buffers.
/// Calls complete before returning, including uploads/downloads, so host and
/// device buffers can safely be released or used in another context afterward.
/// Calls on one context are serialized. Different contexts can run
/// concurrently; callers must not concurrently write the same image through
/// different contexts. No CUDA headers, OpenCV, NPP or cuDNN are required by
/// this public API.
class VIAME_IMAGE_KERNELS_CUDA_EXPORT context {
public:
  explicit context(int device = 0);
  ~context();
  context(context const &) = delete;
  context &operator=(context const &) = delete;
  image allocate(int width, int height, int channels, pixel_type type);
  /// Host rows must be interleaved and contain at least image.row_bytes()
  /// bytes. A zero row stride means tightly packed rows. Host pointers must
  /// remain valid for the duration of the call and span every row described by
  /// the image.
  void upload(image const &destination, void const *host,
              std::size_t row_stride = 0);
  void download(image const &source, void *host, std::size_t row_stride = 0);
  /// Float32, reflect-101 borders, odd size in [1,255], finite sigma >= 0.
  /// Supply output to reuse an allocation. Input/output aliasing is supported.
  image gaussian_blur(image const &source, int size, double sigma = 0,
                      image const *output = nullptr);
  /// Uint8, 1--3 channels, reflect-101 borders. Patch/search sizes in [1,63]
  /// are forced odd as in the CPU API. Strength must be finite and nonnegative.
  /// This is the plain NLM operation, not the Lab-based colored variant.
  image denoise_non_local_means(image const &source, double strength,
                                int patch = 7, int window = 21,
                                image const *output = nullptr);

private:
  struct implementation;
  std::unique_ptr<implementation> impl_;
};

} // namespace cuda
} // namespace image_kernels
} // namespace viame
#endif
