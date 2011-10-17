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
#include "video_reader_process.h"
#include "warp_image_process.h"

#include <vistk/pipeline/process_registry.h>

#include <boost/make_shared.hpp>

/**
 * \file image/registration.cxx
 *
 * \brief Register processes for use.
 */

using namespace vistk;

static process_t create_crop_image_process(config_t const& config);
static process_t create_grayscale_process(config_t const& config);
static process_t create_image_reader_process(config_t const& config);
static process_t create_image_writer_process(config_t const& config);
static process_t create_video_reader_process(config_t const& config);
static process_t create_image_source_process(config_t const& config);
static process_t create_warp_image_process(config_t const& config);

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
  registry->register_process("video_reader", "Reads images from a video.", create_video_reader_process);
  registry->register_process("image_source", "Reads images using different sources.", create_image_source_process);
  registry->register_process("warp_image", "Warps images using tranformation matrices.", create_warp_image_process);

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

process_t
create_video_reader_process(config_t const& config)
{
  return boost::make_shared<video_reader_process>(config);
}

process_t
create_image_source_process(config_t const& config)
{
  static config::key_t const type_key = config::key_t("type");
  static config::value_t const image_list_type = config::value_t("list");
  static config::value_t const vidl_type = config::value_t("vidl");
  static config::value_t const& default_type = image_list_type;

  config::value_t const type_value = config->get_value<config::value_t>(type_key, default_type);

  if (type_value == image_list_type)
  {
    return create_image_reader_process(config);
  }
  else if (type_value == vidl_type)
  {
    return create_video_reader_process(config);
  }

  /// \todo Throw an exception.

  return process_t();
}

process_t
create_warp_image_process(config_t const& config)
{
  return boost::make_shared<warp_image_process>(config);
}
