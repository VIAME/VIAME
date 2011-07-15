/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include <vistk/pipeline/edge_registry.h>

using namespace vistk;

void
register_edges()
{
  edge_registry_t const registry = edge_registry::self();
}
