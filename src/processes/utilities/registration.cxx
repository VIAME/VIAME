/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "homography_reader_process.h"
#include "invert_transform_process.h"
#include "timestamp_reader_process.h"
#include "timestamper_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_registry.h>

/**
 * \file utilities/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

static process_t create_timestamp_source(config_t const& config);

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("utilities_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("homography_reader", "Read homographies from a file", create_process<homography_reader_process>);
  registry->register_process("invert_transform", "Invert tranformation matrices", create_process<invert_transform_process>);
  registry->register_process("timestamp_reader", "Read timestamps from a file", create_process<timestamp_reader_process>);
  registry->register_process("timestamp_source", "A source of timestamps", create_timestamp_source);
  registry->register_process("timestamper", "Generates timestamps", create_process<timestamper_process>);

  registry->mark_module_as_loaded(module_name);
}

process_t
create_timestamp_source(config_t const& config)
{
  static config::key_t const type_key = config::key_t("type");
  static config::value_t const file_type = config::value_t("list");
  static config::value_t const generate_type = config::value_t("generate");
  static config::value_t const& default_type = file_type;

  config::value_t const type_value = config->get_value<config::value_t>(type_key, default_type);

  if (type_value == file_type)
  {
    return create_process<timestamp_reader_process>(config);
  }
  else if (type_value == generate_type)
  {
    return create_process<timestamper_process>(config);
  }

  /// \todo Throw an exception.

  return process_t();
}
