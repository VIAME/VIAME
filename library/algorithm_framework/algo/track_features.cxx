// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief track_feature algorithm instantiation

#include <viame/algorithm_framework/algo/track_features.h>

namespace kwiver {

namespace vital {

namespace algo {

track_features
::track_features()
{
  attach_logger( "algo.track_features" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
