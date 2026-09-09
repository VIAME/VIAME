// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief detect_features algorithm definition instantiation

#include <viame/algorithm_framework/algo/detect_features.h>

namespace kwiver {

namespace vital {

namespace algo {

detect_features
::detect_features()
{
  attach_logger( "algo.detect_features" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
