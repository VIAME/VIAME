/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "read_video_gravl_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_registry.h>

/**
 * \file gravl/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("gravl_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("read_video_gravl", "Read video from a GRAVL resource", create_process<read_video_gravl_process>);

  registry->mark_module_as_loaded(module_name);
}
