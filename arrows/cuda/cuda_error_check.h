// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header file for CUDA error checking helpers
*/

#ifndef KWIVER_ARROWS_CUDA_CUDA_ERROR_CHECK_H_
#define KWIVER_ARROWS_CUDA_CUDA_ERROR_CHECK_H_

#include <cuda_runtime.h>

namespace kwiver {
namespace arrows {
namespace cuda {

/// Macro called to catch cuda error when cuda functions are called
#define CudaErrorCheck(ans) { kwiver::arrows::cuda::cuda_throw((ans), __FILE__, __LINE__); }

void cuda_throw(cudaError_t code, const char *file, int line);

}  // end namespace cuda
}  // end namespace arrows
}  // end namespace kwiver

#endif
