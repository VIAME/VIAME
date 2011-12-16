/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "homography_reader_process.h"
#include "timestamper_process.h"

#include <vistk/pipeline/process_registry.h>

/**
 * \file utilities/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("utilities_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("homography_reader", "A process which reads homographies from a file", create_process<homography_reader_process>);
  registry->register_process("timestamper", "A process which generates timestamps", create_process<timestamper_process>);

  registry->mark_module_as_loaded(module_name);
}
