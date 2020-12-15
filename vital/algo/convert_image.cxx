// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>
#include "convert_image.h"

INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::convert_image);

namespace kwiver {
namespace vital {
namespace algo {

convert_image
::convert_image()
{
  attach_logger( "algo.convert_image" );
}

/// Set this algorithm's properties via a config block
void
convert_image
::set_configuration(kwiver::vital::config_block_sptr config)
{
  (void) config;
}

/// Check that the algorithm's current configuration is valid
bool
convert_image
::check_configuration(kwiver::vital::config_block_sptr config) const
{
  (void) config;
  return true;
}

} } } // end namespace
