// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for KPF exceptions
 */

#include "kpf_exception.h"

#include <sstream>

namespace kwiver {
namespace vital {

kpf_exception
::kpf_exception() noexcept
{
  m_what = "Generic KPF exception";
}

kpf_exception
::~kpf_exception() noexcept
{
}

// ------------------------------------------------------------------
kpf_token_underrun_exception
::kpf_token_underrun_exception(std::string const& message) noexcept
  : m_message(message)
{
  m_what = message;
}

kpf_token_underrun_exception
::~kpf_token_underrun_exception() noexcept
{
}

} // ...vital
} // ...kwiver
