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
