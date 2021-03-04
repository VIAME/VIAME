// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "orphan_cluster.h"

/**
 * \file orphan_cluster.cxx
 *
 * \brief Implementation of the orphan cluster.
 */

namespace sprokit
{

orphan_cluster
::orphan_cluster(kwiver::vital::config_block_sptr const& config)
  : process_cluster(config)
{
}

orphan_cluster
::~orphan_cluster()
{
}

}
