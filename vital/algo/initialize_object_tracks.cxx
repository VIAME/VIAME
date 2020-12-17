// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>
#include <vital/algo/initialize_object_tracks.h>

namespace kwiver {
namespace vital {
namespace algo {

initialize_object_tracks
::initialize_object_tracks()
{
  attach_logger( "algo.initialize_object_tracks" );
}

} } }

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::initialize_object_tracks);
