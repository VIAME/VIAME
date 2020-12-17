// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>

#include "merge_images.h"

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::merge_images);

namespace kwiver {
namespace vital {
namespace algo {

merge_images
::merge_images()
{
  attach_logger( "merge_images" );
}

} } } // end namespace
