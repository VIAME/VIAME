// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>

#include "split_image.h"

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::split_image);

namespace kwiver {
namespace vital {
namespace algo {

split_image
::split_image()
{
  attach_logger( "algo.split_image" );
}

} } } // end namespace
