/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "blocking_pipe_edge.h"
#include "dumb_pipe_edge.h"

#include <vistk/pipeline/edge_registry.h>

using namespace vistk;

static edge_t create_blocking_dumb_pipe(config_t const& config);
static edge_t create_dumb_pipe(config_t const& config);

void
register_edges()
{
  edge_registry_t const registry = edge_registry::self();

  registry->register_edge("blocking_dumb_pipe", "An edge with a capacity", create_blocking_dumb_pipe);
  registry->register_edge("dumb_pipe", "Unlimited-sized edge", create_dumb_pipe);
}

edge_t create_blocking_dumb_pipe(config_t const& config)
{
  return edge_t(new blocking_pipe_edge(config));
}

edge_t create_dumb_pipe(config_t const& config)
{
  return edge_t(new dumb_pipe_edge(config));
}
