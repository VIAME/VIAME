/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_reader_process.h"

#include "vil_helper.h"

#include <vistk/pipeline_types/image_types.h>

#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <fstream>
#include <string>

namespace vistk
{

namespace
{

typedef boost::filesystem::path path_t;

}

class image_reader_process::priv
{
  public:
    priv(path_t const& input_path, read_func_t func);
    ~priv();

    path_t const path;
    read_func_t const read;

    std::ifstream fin;

    bool has_color;

    stamp_t output_stamp;

    static config::key_t const config_pixtype;
    static config::key_t const config_grayscale;
    static config::key_t const config_path;
    static pixtype_t const default_pixtype;
    static bool const default_grayscale;
    static port_t const port_color;
    static port_t const port_output;
};

config::key_t const image_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const image_reader_process::priv::config_grayscale = config::key_t("grayscale");
config::key_t const image_reader_process::priv::config_path = config::key_t("input");
pixtype_t const image_reader_process::priv::default_pixtype = pixtypes::pixtype_byte();
bool const image_reader_process::priv::default_grayscale = false;
process::port_t const image_reader_process::priv::port_color = process::port_t("color");
process::port_t const image_reader_process::priv::port_output = process::port_t("image");

image_reader_process
::image_reader_process(config_t const& config)
  : process(config)
{
  pixtype_t const pixtype = config->get_value<pixtype_t>(priv::config_pixtype, priv::default_pixtype);
  bool const grayscale = config->get_value<bool>(priv::config_grayscale, priv::default_grayscale);
  path_t const path = config->get_value<path_t>(priv::config_path, path_t());

  read_func_t const func = read_for_pixtype(pixtype);

  d = boost::shared_ptr<priv>(new priv(path, func));

  port_type_t const port_type_output = port_type_for_pixtype(pixtype, grayscale);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_color, port_info_t(new port_info(
    type_none,
    port_flags_t(),
    port_description_t("If connected, uses the stamp's color for the output."))));
  declare_output_port(priv::port_output, port_info_t(new port_info(
    port_type_output,
    required,
    port_description_t("The images that are read in."))));

  declare_configuration_key(priv::config_pixtype, conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::default_pixtype),
    config::description_t("The pixel type of the input images."))));
  declare_configuration_key(priv::config_grayscale, conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::default_grayscale),
    config::description_t("Set to \'true\' if the input is grayscale, \'false\' otherwise."))));
  declare_configuration_key(priv::config_path, conf_info_t(new conf_info(
    config::value_t(),
    config::description_t("The input file with a list of images to read."))));
}

image_reader_process
::~image_reader_process()
{
}

void
image_reader_process
::_init()
{
  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  boost::filesystem::path::string_type const path = d->path.native();

  if (path.empty())
  {
    config::value_t const value = config::value_t(path.begin(), path.end());

    throw invalid_configuration_value_exception(name(), priv::config_path, value, "The path given was empty");
  }

  d->fin.open(path.c_str());

  if (!d->fin.good())
  {
    std::string const file_path(path.begin(), path.end());

    throw invalid_configuration_exception(name(), "Failed to open the path: " + file_path);
  }

  if (!input_port_edge(priv::port_color).expired())
  {
    d->has_color = true;
  }

  d->output_stamp = heartbeat_stamp();
}

void
image_reader_process
::_step()
{
  datum_t dat;

  if (d->fin.eof())
  {
    dat = datum::complete_datum();
  }
  else if (!d->fin.good())
  {
    dat = datum::error_datum("Error with input file stream.");
  }
  else
  {
    path_t::string_type line;

    std::getline(d->fin, line);

    dat = d->read(line);
  }

  d->output_stamp = stamp::incremented_stamp(d->output_stamp);

  if (d->has_color)
  {
    edge_datum_t const color_dat = grab_from_port(priv::port_color);

    switch (color_dat.get<0>()->type())
    {
      case datum::DATUM_COMPLETE:
        mark_as_complete();
        dat = datum::complete_datum();
      case datum::DATUM_DATA:
      case datum::DATUM_EMPTY:
        break;
      case datum::DATUM_ERROR:
        dat = datum::error_datum("Error on the color input edge.");
        break;
      case datum::DATUM_INVALID:
      default:
        dat = datum::error_datum("Unrecognized datum type.");
        break;
    }

    d->output_stamp = stamp::recolored_stamp(d->output_stamp, color_dat.get<1>());
  }

  edge_datum_t const edat = edge_datum_t(dat, d->output_stamp);

  push_to_port(priv::port_output, edat);

  process::_step();
}

image_reader_process::priv
::priv(path_t const& input_path, read_func_t func)
  : path(input_path)
  , read(func)
  , has_color(false)
{
}

image_reader_process::priv
::~priv()
{
}

}
