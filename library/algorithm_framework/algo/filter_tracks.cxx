// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Instantiation of \link kwiver::vital::algo::algorithm_def
///        algorithm_def<T> \endlink for
///        \link kwiver::vital::algo::filter_tracks filter_tracks
///        \endlink

#include <viame/algorithm_framework/algo/filter_tracks.h>

namespace kwiver {

namespace vital {

namespace algo {

filter_tracks
::filter_tracks()
{
  attach_logger( "algo.filter_tracks" );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
