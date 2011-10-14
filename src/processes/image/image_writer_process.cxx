/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_writer_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/write.h>

#include <vistk/pipeline_types/image_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/path.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <fstream>
#include <string>

/**
 * \file image_writer_process.cxx
 *
 * \brief Implementation of the image writer process.
 */

namespace vistk
{

class image_writer_process::priv
{
  public:
    priv(path_t const& output_path, config::value_t const& fmt, write_func_t func);
    ~priv();

    typedef boost::basic_format<config::value_t::value_type> format_t;

    path_t const path;
    format_t format;
    write_func_t const write;

    uint64_t count;

    bool has_output;

    std::ofstream fout;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_format;
    static config::key_t const config_path;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_format;
    static config::value_t const default_path;
    static port_t const port_input;
};

config::key_t const image_writer_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const image_writer_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const image_writer_process::priv::config_format = config::key_t("format");
config::key_t const image_writer_process::priv::config_path = config::key_t("output");
config::value_t const image_writer_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const image_writer_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
config::value_t const image_writer_process::priv::default_format = config::value_t("image-%1%-%2%.png");
config::value_t const image_writer_process::priv::default_path = config::value_t("image-%1%.txt");
process::port_t const image_writer_process::priv::port_input = process::port_t("image");

image_writer_process
::image_writer_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    priv::default_pixtype,
    config::description_t("The pixel type of the input images.")));
  declare_configuration_key(priv::config_pixfmt, boost::make_shared<conf_info>(
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images.")));
  declare_configuration_key(priv::config_format, boost::make_shared<conf_info>(
    priv::default_format,
    config::description_t("The format for output filenames.")));
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The input file with a list of images to read.")));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type_input = port_type_for_pixtype(pixtype, pixfmt);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    port_type_input,
    required,
    port_description_t("The images that are to be written.")));
}

image_writer_process
::~image_writer_process()
{
}

void
image_writer_process
::_init()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
    config::value_t const format = config_value<config::value_t>(priv::config_format);
    config::value_t const path_fmt = config_value<config::value_t>(priv::config_path);

    path_t path = path_fmt;

    try
    {
      path = boost::str(priv::format_t(path_fmt) % name());
    }
    catch (boost::io::format_error&)
    {
    }

    write_func_t const func = write_for_pixtype(pixtype);

    d.reset(new priv(path, format, func));
  }

  if (!d->write)
  {
    static std::string const reason = "A write function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  boost::filesystem::path::string_type const path = d->path.native();

  if (path.empty())
  {
    d->has_output = false;
  }
  else
  {
    d->has_output = true;

    d->fout.open(path.c_str());

    if (!d->fout.good())
    {
      std::string const file_path(path.begin(), path.end());

      throw invalid_configuration_exception(name(), "Failed to open the path: " + file_path);
    }
  }
}

void
image_writer_process
::_step()
{
  datum_t const input = grab_datum_from_port(priv::port_input);

  d->format.clear();

  try
  {
    d->format % name();
    d->format % d->count;
  }
  catch (boost::io::format_error&)
  {
  }

  ++d->count;

  path_t const path = boost::str(d->format);

  if (d->has_output)
  {
    path_t::string_type const fstr = path.native();
    std::string const str(fstr.begin(), fstr.end());

    d->fout << str << std::endl;
  }

  d->write(path, input);

  process::_step();
}

image_writer_process::priv
::priv(path_t const& output_path, config::value_t const& fmt, write_func_t func)
  : path(output_path)
  , format(fmt)
  , write(func)
  , count(0)
  , has_output(false)
{
}

image_writer_process::priv
::~priv()
{
}

}
