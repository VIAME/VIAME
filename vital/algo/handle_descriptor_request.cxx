// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/algo/algorithm.txx>

#include "handle_descriptor_request.h"

INSTANTIATE_ALGORITHM_DEF( kwiver::vital::algo::handle_descriptor_request );

namespace kwiver {
namespace vital {
namespace algo {

handle_descriptor_request
::handle_descriptor_request()
{
  attach_logger( "algo.handle_descriptor_request" );
}

/// Set this algorithm's properties via a config block
void
handle_descriptor_request
::set_configuration( kwiver::vital::config_block_sptr config )
{
  (void) config;
}

/// Check that the algorithm's current configuration is valid
bool
handle_descriptor_request
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  (void) config;
  return true;
}

} } } // end namespace
