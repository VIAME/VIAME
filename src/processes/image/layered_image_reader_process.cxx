/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "layered_image_reader_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/read.h>

#include <vistk/utilities/timestamp.h>
#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <string>

/**
 * \file layered_image_reader_process.cxx
 *
 * \brief Implementation of the layered image reader process.
 */

namespace vistk
{

class layered_image_reader_process::priv
{
  public:
    typedef port_t layer_t;
    typedef std::vector<layer_t> layers_t;

    priv(port_type_t const& port_type);
    priv(config::value_t const& fmt, read_func_t func, port_type_t const& port_type, layers_t const& layers_);
    ~priv();

    typedef boost::basic_format<config::value_t::value_type> format_t;

    format_t format;
    read_func_t const read;
    port_type_t const port_type_output;

    layers_t layers;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_format;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_format;
    static port_t const port_timestamp;
    static port_t const port_image_prefix;
};

config::key_t const layered_image_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const layered_image_reader_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const layered_image_reader_process::priv::config_format = config::key_t("format");
config::value_t const layered_image_reader_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const layered_image_reader_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
config::value_t const layered_image_reader_process::priv::default_format = config::value_t("image-%1%-%2%.png");
process::port_t const layered_image_reader_process::priv::port_timestamp = process::port_t("timestamp");
process::port_t const layered_image_reader_process::priv::port_image_prefix = process::port_t("image/");

layered_image_reader_process
::layered_image_reader_process(config_t const& config)
  : process(config)
{
  /// \todo There should probably be timestamp input rather than manually keeping track of the frame.

  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    priv::default_pixtype,
    config::description_t("The pixel type of the input images.")));
  declare_configuration_key(priv::config_pixfmt, boost::make_shared<conf_info>(
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images.")));
  declare_configuration_key(priv::config_format, boost::make_shared<conf_info>(
    priv::default_format,
    config::description_t("The format string for the input layers.")));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type = port_type_for_pixtype(pixtype, pixfmt);

  d.reset(new priv(port_type));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_timestamp, boost::make_shared<port_info>(
    "timestamp",
    required,
    port_description_t("The timestamp to use when reading masks.")));
}

layered_image_reader_process
::~layered_image_reader_process()
{
}

void
layered_image_reader_process
::_init()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
    config::value_t const format = config_value<config::value_t>(priv::config_format);

    read_func_t const func = read_for_pixtype(pixtype);

    d.reset(new priv(format, func, d->port_type_output, d->layers));
  }

  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_init();
}

void
layered_image_reader_process
::_step()
{
  timestamp const ts = grab_from_port_as<timestamp>(priv::port_timestamp);

  BOOST_FOREACH (priv::layer_t const& layer, d->layers)
  {
    d->format.clear();

    try
    {
      d->format % layer;
      d->format % ts.frame();
    }
    catch (boost::io::format_error&)
    {
    }

    path_t const path = boost::str(d->format);

    datum_t const dat = d->read(path);

    push_datum_to_port(priv::port_image_prefix + layer, dat);
  }

  process::_step();
}

process::port_info_t
layered_image_reader_process
::_output_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_image_prefix))
  {
    priv::layer_t const layer = port.substr(priv::port_image_prefix.size());

    priv::layers_t::const_iterator const i = std::find(d->layers.begin(), d->layers.end(), layer);

    if (i == d->layers.end())
    {
      d->layers.push_back(layer);

      port_flags_t required;

      required.insert(flag_required);

      declare_output_port(port, boost::make_shared<port_info>(
        d->port_type_output,
        required,
        port_description_t("The \'" + layer + "\' layer of the image.")));
    }
  }

  return process::_output_port_info(port);
}

layered_image_reader_process::priv
::priv(port_type_t const& port_type)
  : port_type_output(port_type)
{
}

layered_image_reader_process::priv
::priv(config::value_t const& fmt, read_func_t func, port_type_t const& port_type, layers_t const& layers_)
  : format(fmt)
  , read(func)
  , port_type_output(port_type)
  , layers(layers_)
{
}

layered_image_reader_process::priv
::~priv()
{
}

}
