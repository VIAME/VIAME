// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of matlab exceptions
 */

#include "matlab_exception.h"

namespace kwiver {
namespace arrows {
namespace matlab {

matlab_exception::
matlab_exception(const std::string& msg) noexcept
  : vital_exception()
{
    m_what = msg;
}

matlab_exception::
~matlab_exception() noexcept
{ }

} } } // end namespace
