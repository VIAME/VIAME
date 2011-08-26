/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_writer_process.h"

#include "image_helper.h"

#include <vistk/pipeline_types/image_types.h>

#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>

#include <fstream>

namespace vistk
{

namespace
{

typedef boost::filesystem::path path_t;

}

class image_writer_process::priv
{
  public:
    priv(path_t const& output_path, config::value_t const& fmt, write_func_t func);
    ~priv();

    typedef boost::basic_format<path_t::string_type::value_type> format_t;

    path_t const path;
    format_t format;
    write_func_t const write;

    uint64_t count;

    bool has_output;

    std::ofstream fout;

    static config::key_t const config_pixtype;
    static config::key_t const config_grayscale;
    static config::key_t const config_format;
    static config::key_t const config_path;
    static pixtype_t const default_pixtype;
    static bool const default_grayscale;
    static path_t::string_type const default_format;
    static path_t::string_type const default_path;
    static port_t const port_input;
};

config::key_t const image_writer_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const image_writer_process::priv::config_grayscale = config::key_t("grayscale");
config::key_t const image_writer_process::priv::config_format = config::key_t("format");
config::key_t const image_writer_process::priv::config_path = config::key_t("output");
pixtype_t const image_writer_process::priv::default_pixtype = pixtypes::pixtype_byte();
bool const image_writer_process::priv::default_grayscale = false;
path_t::string_type const image_writer_process::priv::default_format = path_t::string_type("image-%1%-%2%.png");
path_t::string_type const image_writer_process::priv::default_path = path_t::string_type("image-%1%.txt");
process::port_t const image_writer_process::priv::port_input = process::port_t("image");

image_writer_process
::image_writer_process(config_t const& config)
  : process(config)
{
  pixtype_t const pixtype = config->get_value<pixtype_t>(priv::config_pixtype, priv::default_pixtype);
  bool const grayscale = config->get_value<bool>(priv::config_grayscale, priv::default_grayscale);
  path_t::string_type const format = config->get_value<path_t::string_type>(priv::config_format, priv::default_format);
  path_t::string_type const path_fmt = config->get_value<path_t::string_type>(priv::config_path, priv::default_path);

  path_t path = path_fmt;

  try
  {
    path = boost::str(priv::format_t(path_fmt) % name());
  }
  catch (boost::io::format_error&)
  {
  }

  write_func_t const func = write_for_pixtype(pixtype);

  d = boost::make_shared<priv>(path, format, func);

  port_type_t const port_type_input = port_type_for_pixtype(pixtype, grayscale);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    port_type_input,
    required,
    port_description_t("The images that are to be written.")));

  declare_configuration_key(priv::config_pixtype, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_pixtype),
    config::description_t("The pixel type of the input images.")));
  declare_configuration_key(priv::config_grayscale, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_grayscale),
    config::description_t("Set to \'true\' if the input is grayscale, \'false\' otherwise.")));
  declare_configuration_key(priv::config_format, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_format),
    config::description_t("The format for output filenames.")));
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The input file with a list of images to read.")));
}

image_writer_process
::~image_writer_process()
{
}

void
image_writer_process
::_init()
{
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
  edge_datum_t const input_dat = grab_from_port(priv::port_input);
  datum_t const input_datum = input_dat.get<0>();

  datum_t dat;

  switch (input_datum->type())
  {
    case datum::DATUM_DATA:
    {
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

      d->write(path, input_datum);
      break;
    }
    case datum::DATUM_COMPLETE:
      mark_as_complete();
      break;
    case datum::DATUM_EMPTY:
    case datum::DATUM_ERROR:
    case datum::DATUM_INVALID:
    default:
      break;
  }

  process::_step();
}

image_writer_process::priv
::priv(path_t const& output_path, config::value_t const& fmt, write_func_t func)
  : path(output_path)
  , format(fmt)
  , write(func)
  , count(0)
{
}

image_writer_process::priv
::~priv()
{
}

}
