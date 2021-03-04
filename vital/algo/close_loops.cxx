// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>
#include <vital/algo/close_loops.h>

namespace kwiver {
namespace vital {
namespace algo {

close_loops
::close_loops()
{
  attach_logger( "algo.close_loops" );
}

} } }

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::close_loops);
