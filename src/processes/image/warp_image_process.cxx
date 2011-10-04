/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "warp_image_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/warp.h>

#include <vistk/utilities/homography.h>

#include <vistk/pipeline_types/utility_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/make_shared.hpp>

#include <string>

/**
 * \file warp_image_process.cxx
 *
 * \brief Implementation of the warp image process.
 */

namespace vistk
{

class warp_image_process::priv
{
  public:
    priv(warp_func_t func);
    ~priv();

    warp_func_t const warp;

    static config::key_t const config_pixtype;
    static config::key_t const config_grayscale;
    static config::value_t const default_pixtype;
    static config::value_t const default_grayscale;
    static port_t const port_transform;
    static port_t const port_input;
    static port_t const port_output;
};

config::key_t const warp_image_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const warp_image_process::priv::config_grayscale = config::key_t("grayscale");
config::value_t const warp_image_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const warp_image_process::priv::default_grayscale = config::value_t("false");
process::port_t const warp_image_process::priv::port_transform = port_t("transform");
process::port_t const warp_image_process::priv::port_input = port_t("image");
process::port_t const warp_image_process::priv::port_output = port_t("warped_image");

warp_image_process
::warp_image_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    priv::default_pixtype,
    config::description_t("The pixel type of the input images.")));
  declare_configuration_key(priv::config_grayscale, boost::make_shared<conf_info>(
    priv::default_grayscale,
    config::description_t("Set to \'true\' if the input is grayscale, \'false\' otherwise.")));

  pixtype_t const pixtype = config->get_value<pixtype_t>(priv::config_pixtype, priv::default_pixtype);
  bool const grayscale = config_value<bool>(priv::config_grayscale);

  port_type_t const port_type = port_type_for_pixtype(pixtype, grayscale);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_transform, boost::make_shared<port_info>(
    utility_types::t_transform,
    required,
    port_description_t("The transform to use to warp the image")));
  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    port_type,
    required,
    port_description_t("The image to warp.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    port_type,
    required,
    port_description_t("The warped image.")));
}

warp_image_process
::~warp_image_process()
{
}

void
warp_image_process
::_init()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);

    warp_func_t const func = warp_for_pixtype(pixtype);

    d.reset(new priv(func));
  }

  if (!d->warp)
  {
    static std::string const reason = "A warping function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }
}

void
warp_image_process
::_step()
{
  datum_t const transform = grab_datum_from_port(priv::port_transform);
  datum_t const input = grab_datum_from_port(priv::port_input);

  datum_t const dat = d->warp(input, transform);

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

warp_image_process::priv
::priv(warp_func_t func)
  : warp(func)
{
}

warp_image_process::priv
::~priv()
{
}

}
