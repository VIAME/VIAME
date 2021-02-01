// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header file for CUDA memory management function
*/

#ifndef KWIVER_ARROWS_CUDA_CUDA_MEMORY_H_
#define KWIVER_ARROWS_CUDA_CUDA_MEMORY_H_

#include "cuda_error_check.h"
#include <cuda_runtime.h>
#include <memory>

namespace kwiver {
namespace arrows {
namespace cuda {

/// A CUDA delete functor to use with a unique_ptr
template <typename T>
struct cuda_deleter
{
  constexpr cuda_deleter() noexcept = default;

  void operator() (T* ptr) const
  {
    cudaFree(ptr);
  }
};

/// Provide a short name for the CUDA unique_ptr
template <typename T>
using cuda_ptr = std::unique_ptr<T[], cuda_deleter<T> >;

/// Construct a unique_ptr to CUDA memory
template <typename T>
cuda_ptr<T>
make_cuda_mem(size_t size)
{
  T* ptr;
  CudaErrorCheck(cudaMalloc((void**)&ptr, size * sizeof(T)));
  return cuda_ptr<T>(ptr);
}

}  // end namespace cuda
}  // end namespace arrows
}  // end namespace kwiver

#endif
