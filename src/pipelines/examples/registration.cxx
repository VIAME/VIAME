/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "thread_per_process_pipeline.h"
#include "thread_pool_pipeline.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/pipeline_registry.h>

using namespace vistk;

static pipeline_t create_thread_per_process_pipeline(config_t const& config);
static pipeline_t create_thread_pool_pipeline(config_t const& config);

void
register_pipelines()
{
  pipeline_registry_t const registry = pipeline_registry::self();

  registry->register_pipeline("thread_per_process", "Runs each process in its own thread", create_thread_per_process_pipeline);
  registry->register_pipeline("thread_pool", "Uses a pool of threads to step processes", create_thread_pool_pipeline);
}

pipeline_t
create_thread_per_process_pipeline(config_t const& config)
{
  return pipeline_t(new thread_per_process_pipeline(config));
}

pipeline_t
create_thread_pool_pipeline(config_t const& config)
{
  size_t const num_threads = config->get_value<size_t>("num_threads", 0);

  return pipeline_t(new thread_pool_pipeline(config, num_threads));
}
