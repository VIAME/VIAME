// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for GPU exceptions
 */

#include "gpu.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
gpu_exception
::gpu_exception() noexcept
{
  m_what = "GPU error";
}

gpu_exception
::~gpu_exception() noexcept
{
}

// ------------------------------------------------------------------
gpu_memory_exception
::gpu_memory_exception( std::string const& msg) noexcept
{
  m_what = "GPU memory exception: " + msg;
}

gpu_memory_exception
::~gpu_memory_exception() noexcept
{
}

} } // end namespace
