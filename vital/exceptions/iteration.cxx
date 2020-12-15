// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL Exceptions pertaining to iteration.
 */

#include "iteration.h"

#include <sstream>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
stop_iteration_exception::
stop_iteration_exception( std::string const& container ) noexcept
{
  std::ostringstream ss;

  ss << "Attempt to iterate past the end of a "
     << container << " container.";

  m_what = ss.str();
}

} } // end namespaces
