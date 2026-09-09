// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/algorithm_framework/algo/dynamic_configuration.h>

namespace kwiver {

namespace vital {

namespace algo {

dynamic_configuration
::dynamic_configuration()
{
  attach_logger( "algo.dynamic_configuration" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
