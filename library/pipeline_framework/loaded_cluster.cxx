// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file   loaded_cluster.cxx
 * @brief  Implementation for loaded_cluster class
 */

#include "loaded_cluster.h"

namespace sprokit {

loaded_cluster
::loaded_cluster(kwiver::vital::config_block_sptr const& config)
  : process_cluster(config)
{
}

loaded_cluster
::~loaded_cluster()
{
}

} // end namespace sprokit
