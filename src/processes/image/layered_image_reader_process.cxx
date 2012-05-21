/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "layered_image_reader_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/read.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <fstream>
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
    priv(path_t const& input_path, read_func_t func, port_type_t const& port_type, layers_t const& layers_);
    ~priv();

    typedef boost::basic_format<config::value_t::value_type> format_t;

    path_t path;
    read_func_t const read;
    port_type_t const port_type_output;

    std::ifstream fin;

    layers_t layers;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_path;
    static config::key_t const config_format;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_format;
    static port_t const port_image_prefix;
};

config::key_t const layered_image_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const layered_image_reader_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const layered_image_reader_process::priv::config_path = config::key_t("path");
config::key_t const layered_image_reader_process::priv::config_format = config::key_t("format");
config::value_t const layered_image_reader_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const layered_image_reader_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
config::value_t const layered_image_reader_process::priv::default_format = config::value_t("image-%1%-%2%.png");
process::port_t const layered_image_reader_process::priv::port_image_prefix = process::port_t("image/");

layered_image_reader_process
::layered_image_reader_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    priv::default_pixtype,
    config::description_t("The pixel type of the input images.")));
  declare_configuration_key(priv::config_pixfmt, boost::make_shared<conf_info>(
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images.")));
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The input file with a list of image format paths to read.")));
  declare_configuration_key(priv::config_format, boost::make_shared<conf_info>(
    priv::default_format,
    config::description_t("The format string for the input layers.")));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type = port_type_for_pixtype(pixtype, pixfmt);

  d.reset(new priv(port_type));
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
    path_t const path = config_value<path_t>(priv::config_path);

    read_func_t const func = read_for_pixtype(pixtype);

    d.reset(new priv(path, func, d->port_type_output, d->layers));
  }

  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  path_t::string_type const path = d->path.native();

  if (path.empty())
  {
    static std::string const reason = "The path given was empty";
    config::value_t const value = config::value_t(path.begin(), path.end());

    throw invalid_configuration_value_exception(name(), priv::config_path, value, reason);
  }

  d->fin.open(path.c_str());

  if (!d->fin.good())
  {
    std::string const file_path(path.begin(), path.end());
    std::string const reason = "Failed to open the path: " + file_path;

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
layered_image_reader_process
::_step()
{
  datum_t dat;
  std::string line;
  bool complete = false;

  if (d->fin.eof())
  {
    dat = datum::complete_datum();
    complete = true;
  }
  else if (!d->fin.good())
  {
    static datum::error_t const err_string = datum::error_t("Error with input file stream.");

    dat = datum::error_datum(err_string);
  }
  else
  {
    std::getline(d->fin, line);

    if (line.empty())
    {
      dat = datum::empty_datum();
    }
  }

  priv::format_t fmt = priv::format_t(line);

  BOOST_FOREACH (priv::layer_t const& layer, d->layers)
  {
    port_t const output_port = priv::port_image_prefix + layer;
    datum_t odat;

    if (dat)
    {
      odat = dat;
    }
    else
    {
      fmt.clear();

      try
      {
        fmt % layer;
      }
      catch (boost::io::format_error&)
      {
      }

      path_t const path = boost::str(fmt);

      odat = d->read(path);
    }

    push_datum_to_port(priv::port_image_prefix + layer, odat);
  }

  if (complete)
  {
    mark_process_as_complete();
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
::priv(path_t const& input_path, read_func_t func, port_type_t const& port_type, layers_t const& layers_)
  : path(input_path)
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
