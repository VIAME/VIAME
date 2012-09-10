/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "crop_image_process.h"

#include <processes/helpers/image/crop.h>
#include <processes/helpers/image/format.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <string>

#include <cstddef>

/**
 * \file crop_image_process.cxx
 *
 * \brief Implementation of the image cropping process.
 */

namespace vistk
{

class crop_image_process::priv
{
  public:
    priv(crop_func_t func, size_t x_, size_t y_, size_t w_, size_t h_);
    ~priv();

    crop_func_t const crop;

    size_t x;
    size_t y;
    size_t w;
    size_t h;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_x_offset;
    static config::key_t const config_y_offset;
    static config::key_t const config_width;
    static config::key_t const config_height;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_x_offset;
    static config::value_t const default_y_offset;
    static config::value_t const default_width;
    static config::value_t const default_height;
    static port_t const port_input;
    static port_t const port_output;
};

config::key_t const crop_image_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const crop_image_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const crop_image_process::priv::config_x_offset = config::key_t("x_offset");
config::key_t const crop_image_process::priv::config_y_offset = config::key_t("y_offset");
config::key_t const crop_image_process::priv::config_width = config::key_t("width");
config::key_t const crop_image_process::priv::config_height = config::key_t("height");
config::value_t const crop_image_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const crop_image_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
config::value_t const crop_image_process::priv::default_x_offset = config::value_t("0");
config::value_t const crop_image_process::priv::default_y_offset = config::value_t("0");
config::value_t const crop_image_process::priv::default_width = config::value_t("0");
config::value_t const crop_image_process::priv::default_height = config::value_t("0");
process::port_t const crop_image_process::priv::port_input = port_t("image");
process::port_t const crop_image_process::priv::port_output = port_t("cropimage");

crop_image_process
::crop_image_process(config_t const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_pixtype,
    priv::default_pixtype,
    config::description_t("The pixel type of the input images."));
  declare_configuration_key(
    priv::config_pixfmt,
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images."));
  declare_configuration_key(
    priv::config_x_offset,
    priv::default_x_offset,
    config::description_t("The x offset to start cropping at."));
  declare_configuration_key(
    priv::config_y_offset,
    priv::default_y_offset,
    config::description_t("The y offset to start cropping at."));
  declare_configuration_key(
    priv::config_width,
    priv::default_width,
    config::description_t("The width of the cropped image."));
  declare_configuration_key(
    priv::config_height,
    priv::default_height,
    config::description_t("The height of the cropped image."));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type = port_type_for_pixtype(pixtype, pixfmt);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    port_type,
    required,
    port_description_t("The image to crop."));
  declare_output_port(
    priv::port_output,
    port_type,
    required,
    port_description_t("The resulting cropped image."));
}

crop_image_process
::~crop_image_process()
{
}

void
crop_image_process
::_configure()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
    size_t const x = config_value<size_t>(priv::config_x_offset);
    size_t const y = config_value<size_t>(priv::config_y_offset);
    size_t const w = config_value<size_t>(priv::config_width);
    size_t const h = config_value<size_t>(priv::config_height);

    crop_func_t const func = crop_for_pixtype(pixtype);

    d.reset(new priv(func, x, y, w, h));
  }

  if (!d->crop)
  {
    static std::string const reason = "A cropping function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  /// \todo Check crop dimensions.

  process::_init();
}

void
crop_image_process
::_step()
{
  datum_t const input = grab_datum_from_port(priv::port_input);

  datum_t const dat = d->crop(input, d->x, d->y, d->w, d->h);

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

crop_image_process::priv
::priv(crop_func_t func, size_t x_, size_t y_, size_t w_, size_t h_)
  : crop(func)
  , x(x_)
  , y(y_)
  , w(w_)
  , h(h_)
{
}

crop_image_process::priv
::~priv()
{
}

}
