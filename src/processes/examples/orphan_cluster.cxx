/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "orphan_cluster.h"

/**
 * \file orphan_cluster.cxx
 *
 * \brief Implementation of the orphan cluster.
 */

namespace vistk
{

orphan_cluster
::orphan_cluster(config_t const& config)
  : process_cluster(config)
{
}

orphan_cluster
::~orphan_cluster()
{
}

}
