/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "layered_image_reader_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/read.h>

#include <vistk/utilities/path.h>
#include <vistk/utilities/timestamp.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>

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
    priv(read_func_t func, port_type_t const& port_type, layers_t const& layers_);
    ~priv();

    typedef boost::basic_format<config::value_t::value_type> format_t;

    read_func_t const read;
    port_type_t const port_type_output;

    timestamp::frame_t frame;

    layers_t layers;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static port_t const port_input;
    static port_t const port_image_prefix;
    static port_t const port_timestamp;
};

config::key_t const layered_image_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const layered_image_reader_process::priv::config_pixfmt = config::key_t("pixfmt");
config::value_t const layered_image_reader_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const layered_image_reader_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
process::port_t const layered_image_reader_process::priv::port_input = port_t("path_format");
process::port_t const layered_image_reader_process::priv::port_image_prefix = port_t("image/");
process::port_t const layered_image_reader_process::priv::port_timestamp = port_t("timestamp");

layered_image_reader_process
::layered_image_reader_process(config_t const& config)
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

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type = port_type_for_pixtype(pixtype, pixfmt);

  d.reset(new priv(port_type));

  port_flags_t const none;

  declare_output_port(
    priv::port_timestamp,
    "timestamp",
    none,
    port_description_t("The timestamp for the image."));
}

layered_image_reader_process
::~layered_image_reader_process()
{
}

void
layered_image_reader_process
::_configure()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);

    read_func_t const func = read_for_pixtype(pixtype);

    d.reset(new priv(func, d->port_type_output, d->layers));
  }

  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
layered_image_reader_process
::_step()
{
  path_t const path = grab_from_port_as<path_t>(priv::port_input);

  std::string const str = path.string<std::string>();

  priv::format_t fmt = priv::format_t(str);

  BOOST_FOREACH (priv::layer_t const& layer, d->layers)
  {
    port_t const output_port = priv::port_image_prefix + layer;

    fmt.clear();

    try
    {
      fmt % layer;
    }
    catch (boost::io::format_error const&)
    {
    }

    path_t const formatted_path = boost::str(fmt);

    datum_t const odat = d->read(formatted_path);

    push_datum_to_port(priv::port_image_prefix + layer, odat);
  }

  ++d->frame;

  timestamp const ts = timestamp(d->frame);

  datum_t const dat_ts = datum::new_datum(ts);
  push_datum_to_port(priv::port_timestamp, dat_ts);

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

      declare_output_port(
        port,
        d->port_type_output,
        required,
        port_description_t("The \'" + layer + "\' layer of the image."));
    }
  }

  return process::_output_port_info(port);
}

layered_image_reader_process::priv
::priv(port_type_t const& port_type)
  : read()
  , port_type_output(port_type)
  , layers()
{
}

layered_image_reader_process::priv
::priv(read_func_t func, port_type_t const& port_type, layers_t const& layers_)
  : read(func)
  , port_type_output(port_type)
  , frame(0)
  , layers(layers_)
{
}

layered_image_reader_process::priv
::~priv()
{
}

}
