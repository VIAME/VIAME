/*ckwg +29
* Copyright 2019 by Kitware, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
*  * Neither name of Kitware, Inc. nor the names of any contributors may be used
*    to endorse or promote products derived from this software without specific
*    prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
