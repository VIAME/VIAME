// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of canonical similarity transform
///        estimation algorithm definition.

#include <viame/algorithm_framework/algo/estimate_canonical_transform.h>

namespace kwiver {

namespace vital {

namespace algo {

estimate_canonical_transform
::estimate_canonical_transform()
{
  attach_logger( "algo.estimate_canonical_transform" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
