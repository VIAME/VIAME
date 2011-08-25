/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "registration.h"

#include "crop_image_process.h"
#include "grayscale_process.h"
#include "image_reader_process.h"
#include "image_writer_process.h"

#include <vistk/pipeline/process_registry.h>

#include <boost/make_shared.hpp>

using namespace vistk;

static process_t create_crop_image_process(config_t const& config);
static process_t create_grayscale_process(config_t const& config);
static process_t create_image_reader_process(config_t const& config);
static process_t create_image_writer_process(config_t const& config);

void
register_processes()
{
  static process_registry::module_t const module_name = process_registry::module_t("image_processes");

  process_registry_t const registry = process_registry::self();

  if (registry->is_module_loaded(module_name))
  {
    return;
  }

  registry->register_process("crop_image", "Crop an image to a specific size.", create_crop_image_process);
  registry->register_process("grayscale", "Convert an RGB image into grayscale.", create_grayscale_process);
  registry->register_process("image_reader", "Read images from files given a list of images.", create_image_reader_process);
  registry->register_process("image_writer", "Write images to files.", create_image_writer_process);

  registry->mark_module_as_loaded(module_name);
}

process_t
create_crop_image_process(config_t const& config)
{
  return boost::make_shared<crop_image_process>(config);
}

process_t
create_grayscale_process(config_t const& config)
{
  return boost::make_shared<grayscale_process>(config);
}

process_t
create_image_reader_process(config_t const& config)
{
  return boost::make_shared<image_reader_process>(config);
}

process_t
create_image_writer_process(config_t const& config)
{
  return boost::make_shared<image_writer_process>(config);
}
