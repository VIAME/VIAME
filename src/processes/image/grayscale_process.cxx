/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "grayscale_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/grayscale.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/make_shared.hpp>

#include <string>

/**
 * \file grayscale_process.cxx
 *
 * \brief Implementation of the grayscale process.
 */

namespace vistk
{

class grayscale_process::priv
{
  public:
    priv(gray_func_t func);
    ~priv();

    gray_func_t const convert;

    static config::key_t const config_pixtype;
    static config::value_t const default_pixtype;
    static port_t const port_input;
    static port_t const port_output;
};

config::key_t const grayscale_process::priv::config_pixtype = config::key_t("pixtype");
config::value_t const grayscale_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
process::port_t const grayscale_process::priv::port_input = port_t("rgbimage");
process::port_t const grayscale_process::priv::port_output = port_t("grayimage");

grayscale_process
::grayscale_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    priv::default_pixtype,
    config::description_t("The pixel type of the input images.")));

  pixtype_t const pixtype = config->get_value<pixtype_t>(priv::config_pixtype, priv::default_pixtype);

  port_type_t const port_type_input = port_type_for_pixtype(pixtype, false);
  port_type_t const port_type_output = port_type_for_pixtype(pixtype, true);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    port_type_input,
    required,
    port_description_t("The image to turn into grayscale.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    port_type_output,
    required,
    port_description_t("The resulting grayscale image.")));
}

grayscale_process
::~grayscale_process()
{
}

void
grayscale_process
::_init()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);

    gray_func_t const func = gray_for_pixtype(pixtype);

    d.reset(new priv(func));
  }

  if (!d->convert)
  {
    static std::string const reason = "A conversion function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }
}

void
grayscale_process
::_step()
{
  datum_t const input = grab_datum_from_port(priv::port_input);

  datum_t const dat = d->convert(input);

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

grayscale_process::priv
::priv(gray_func_t func)
  : convert(func)
{
}

grayscale_process::priv
::~priv()
{
}

}
