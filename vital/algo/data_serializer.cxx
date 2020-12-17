// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief data_serializer algorithm definition instantiation
 */

#include <vital/algo/data_serializer.h>
#include <vital/algo/algorithm.txx>
#include <vital/util/string.h>

#include <sstream>
#include <stdexcept>
#include <vector>

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------------------
data_serializer
::data_serializer()
{
  attach_logger( "data_serializer" );
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::data_serializer);
/// \endcond
