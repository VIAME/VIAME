// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Instantiation of \link kwiver::vital::algo::algorithm_def algorithm_def<T>
 *        \endlink for \link kwiver::vital::algo::filter_tracks
 *        filter_tracks \endlink
 */

#include <vital/algo/filter_tracks.h>
#include <vital/algo/algorithm.txx>

namespace kwiver {
namespace vital {
namespace algo {

filter_tracks
::filter_tracks()
{
  attach_logger( "algo.filter_tracks" );
}

} } } // end namespace

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::filter_tracks);
