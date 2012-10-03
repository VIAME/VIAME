/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_reader_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/read.h>

#include <vistk/utilities/path.h>
#include <vistk/utilities/timestamp.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/fstream.hpp>

#include <string>

/**
 * \file image_reader_process.cxx
 *
 * \brief Implementation of the image reader process.
 */

namespace vistk
{

class image_reader_process::priv
{
  public:
    priv(path_t const& input_path, read_func_t func, bool ver);
    ~priv();

    path_t const path;
    read_func_t const read;
    bool const verify;

    timestamp::frame_t frame;

    boost::filesystem::ifstream fin;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_path;
    static config::key_t const config_verify;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_verify;
    static port_t const port_output;
    static port_t const port_output_ts;
};

config::key_t const image_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const image_reader_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const image_reader_process::priv::config_path = config::key_t("input");
config::key_t const image_reader_process::priv::config_verify = config::key_t("verify");
config::value_t const image_reader_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const image_reader_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
config::value_t const image_reader_process::priv::default_verify = config::value_t("false");
process::port_t const image_reader_process::priv::port_output = port_t("image");
process::port_t const image_reader_process::priv::port_output_ts = port_t("timestamp");

image_reader_process
::image_reader_process(config_t const& config)
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
    priv::config_path,
    config::value_t(),
    config::description_t("The input file with a list of images to read."));
  declare_configuration_key(
    priv::config_verify,
    priv::default_verify,
    config::description_t("If \'true\', the paths in the input file will checked that they can be read."));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type_output = port_type_for_pixtype(pixtype, pixfmt);

  port_flags_t const none;
  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(
    priv::port_output,
    port_type_output,
    required,
    port_description_t("The images that are read in."));
  declare_output_port(
    priv::port_output_ts,
    "timestamp",
    none,
    port_description_t("The timestamp for the image."));
}

image_reader_process
::~image_reader_process()
{
}

void
image_reader_process
::_configure()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
    path_t const path = config_value<path_t>(priv::config_path);
    bool const verify = config_value<bool>(priv::config_verify);

    read_func_t const func = read_for_pixtype(pixtype);

    d.reset(new priv(path, func, verify));
  }

  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  if (d->path.empty())
  {
    static std::string const reason = "The path given was empty";
    path_t::string_type const& pstr = d->path.native();
    config::value_t const value = config::value_t(pstr.begin(), pstr.end());

    throw invalid_configuration_value_exception(name(), priv::config_path, value, reason);
  }

  d->fin.open(d->path);

  if (!d->fin.good())
  {
    path_t::string_type const& pstr = d->path.native();
    std::string const str(pstr.begin(), pstr.end());
    std::string const reason = "Failed to open the path: " + str;

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
image_reader_process
::_init()
{
  if (d->verify)
  {
    while (d->fin.good())
    {
      std::string line;

      std::getline(d->fin, line);

      if (line.empty())
      {
        continue;
      }

      datum_t dat = d->read(line);

      if (dat->type() != datum::data)
      {
        std::string const reason = "The file \'" + line + "\' could not be read";

        throw invalid_configuration_exception(name(), reason);
      }
    }

    d->fin.clear();
    d->fin.seekg(0, std::ios::beg);
  }

  process::_init();
}

void
image_reader_process
::_step()
{
  datum_t dat;
  datum_t dat_ts;

  if (d->fin.eof())
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
    dat_ts = dat;
  }
  else if (!d->fin.good())
  {
    static datum::error_t const err_string = datum::error_t("Error with input file stream.");

    dat = datum::error_datum(err_string);
    dat_ts = dat;
  }
  else
  {
    std::string line;

    std::getline(d->fin, line);

    if (line.empty())
    {
      dat = datum::empty_datum();
    }
    else
    {
      dat = d->read(line);
    }

    ++d->frame;

    timestamp const ts = timestamp(d->frame);

    dat_ts = datum::new_datum(ts);
  }

  push_datum_to_port(priv::port_output, dat);
  push_datum_to_port(priv::port_output_ts, dat_ts);

  process::_step();
}

image_reader_process::priv
::priv(path_t const& input_path, read_func_t func, bool ver)
  : path(input_path)
  , read(func)
  , verify(ver)
  , frame(0)
  , fin()
{
}

image_reader_process::priv
::~priv()
{
}

}
