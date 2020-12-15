// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/dynamic_configuration.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

dynamic_configuration
::dynamic_configuration()
{
  attach_logger( "algo.dynamic_configuration" );
}

} } }

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::dynamic_configuration);
/// \endcond
