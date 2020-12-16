// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Implementation of CUDA error checking helpers
*/

#include "cuda_error_check.h"

#include <vital/exceptions/gpu.h>
#include <vital/logger/logger.h>
#include <kwiversys/SystemTools.hxx>
#include <cuda_runtime.h>

namespace kwiver {
namespace arrows {
namespace cuda {

void cuda_throw(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    auto filename = kwiversys::SystemTools::GetFilenameName(file);
    auto basename = kwiversys::SystemTools::GetFilenameWithoutExtension(filename);
    auto logger = vital::get_logger("arrows.cuda."+basename);
    LOG_ERROR(logger, "GPU Error: " << cudaGetErrorString(code)
      << "\n  in " << filename << ":" << line);
    auto e = vital::gpu_memory_exception(cudaGetErrorString(code));
    e.set_location(filename.c_str(), line);
    throw e;
  }
}

}  // end namespace cuda
}  // end namespace arrows
}  // end namespace kwiver
