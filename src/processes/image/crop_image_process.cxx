/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "crop_image_process.h"

#include "vil_helper.h"

#include <vistk/pipeline/process_exception.h>

#include <boost/make_shared.hpp>

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
    static config::key_t const config_grayscale;
    static config::key_t const config_x_offset;
    static config::key_t const config_y_offset;
    static config::key_t const config_width;
    static config::key_t const config_height;
    static config::value_t const default_pixtype;
    static bool const default_grayscale;
    static size_t const default_x_offset;
    static size_t const default_y_offset;
    static size_t const default_width;
    static size_t const default_height;
    static port_t const port_input;
    static port_t const port_output;
};

config::key_t const crop_image_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const crop_image_process::priv::config_grayscale = config::key_t("grayscale");
config::key_t const crop_image_process::priv::config_x_offset = config::key_t("x_offset");
config::key_t const crop_image_process::priv::config_y_offset = config::key_t("y_offset");
config::key_t const crop_image_process::priv::config_width = config::key_t("width");
config::key_t const crop_image_process::priv::config_height = config::key_t("height");
pixtype_t const crop_image_process::priv::default_pixtype = pixtypes::pixtype_byte();
bool const crop_image_process::priv::default_grayscale = false;
size_t const crop_image_process::priv::default_x_offset = 0;
size_t const crop_image_process::priv::default_y_offset = 0;
size_t const crop_image_process::priv::default_width = 0;
size_t const crop_image_process::priv::default_height = 0;
process::port_t const crop_image_process::priv::port_input = port_t("image");
process::port_t const crop_image_process::priv::port_output = port_t("cropimage");

crop_image_process
::crop_image_process(config_t const& config)
  : process(config)
{
  pixtype_t const pixtype = config->get_value<pixtype_t>(priv::config_pixtype, priv::default_pixtype);
  bool const grayscale = config->get_value<bool>(priv::config_grayscale, priv::default_grayscale);
  size_t const x = config->get_value<size_t>(priv::config_x_offset, priv::default_x_offset);
  size_t const y = config->get_value<size_t>(priv::config_y_offset, priv::default_y_offset);
  size_t const w = config->get_value<size_t>(priv::config_width, priv::default_width);
  size_t const h = config->get_value<size_t>(priv::config_height, priv::default_height);

  crop_func_t const func = crop_for_pixtype(pixtype);

  d = boost::make_shared<priv>(func, x, y, w, h);

  port_type_t const port_type = port_type_for_pixtype(pixtype, grayscale);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    port_type,
    required,
    port_description_t("The image to turn into grayscale.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    port_type,
    required,
    port_description_t("The resulting grayscale image.")));

  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_pixtype),
    config::description_t("The pixel type of the input images.")));
  declare_configuration_key(priv::config_x_offset, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_x_offset),
    config::description_t("The x offset to start cropping at.")));
  declare_configuration_key(priv::config_y_offset, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_y_offset),
    config::description_t("The y offset to start cropping at.")));
  declare_configuration_key(priv::config_width, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_width),
    config::description_t("The width of the cropped image.")));
  declare_configuration_key(priv::config_height, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_height),
    config::description_t("The height of the cropped image.")));
}

crop_image_process
::~crop_image_process()
{
}

void
crop_image_process
::_init()
{
  if (!d->crop)
  {
    static std::string const reason = "A cropping function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }
}

void
crop_image_process
::_step()
{
  edge_datum_t const input_dat = grab_from_port(priv::port_input);
  datum_t const input_datum = input_dat.get<0>();
  stamp_t const input_stamp = input_dat.get<1>();

  datum_t dat;

  switch (input_datum->type())
  {
    case datum::DATUM_DATA:
    {
      dat = d->crop(input_datum, d->x, d->y, d->w, d->h);
      break;
    }
    case datum::DATUM_EMPTY:
      dat = datum::empty_datum();
      break;
    case datum::DATUM_COMPLETE:
      dat = datum::complete_datum();
      mark_as_complete();
      break;
    case datum::DATUM_ERROR:
      dat = datum::error_datum("Error on the input edges.");
      break;
    case datum::DATUM_INVALID:
    default:
      dat = datum::error_datum("Unrecognized datum type.");
      break;
  }

  edge_datum_t const edat = edge_datum_t(dat, input_stamp);

  push_to_port(priv::port_output, edat);

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
