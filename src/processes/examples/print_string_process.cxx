/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "print_string_process.h"

#include <vistk/pipeline_types/port_types.h>

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/path.hpp>

#include <iostream>
#include <fstream>
#include <string>

namespace vistk
{

class print_string_process::priv
{
  public:
    typedef std::string string_t;
    typedef boost::filesystem::path path_t;

    priv(path_t const& output_path);
    ~priv();

    path_t const path;

    std::ofstream fout;

    static config::key_t const CONFIG_PATH_NAME;
    static port_t const INPUT_PORT_NAME;
};

config::key_t const print_string_process::priv::CONFIG_PATH_NAME = config::key_t("output");
process::port_t const print_string_process::priv::INPUT_PORT_NAME = process::port_t("string");

print_string_process
::print_string_process(config_t const& config)
  : process(config)
{
  priv::path_t path = config->get_value<priv::path_t>(priv::CONFIG_PATH_NAME, priv::path_t());

  d = boost::shared_ptr<priv>(new priv(path));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::INPUT_PORT_NAME, port_info_t(new port_info(
    port_types::t_string,
    required,
    port_description_t("Where strings are read from."))));

  declare_configuration_key(priv::CONFIG_PATH_NAME, conf_info_t(new conf_info(
    config::value_t(),
    config::description_t("The path of the file to output to."))));
}

print_string_process
::~print_string_process()
{
}

void
print_string_process
::_init()
{
  boost::filesystem::path::string_type const path = d->path.native();

  if (path.empty())
  {
    config::value_t const value = config::value_t(path.begin(), path.end());

    throw invalid_configuration_value_exception(name(), priv::CONFIG_PATH_NAME, value, "The path given was empty");
  }

  d->fout.open(path.c_str());

  if (!d->fout.good())
  {
    std::string const file_path(path.begin(), path.end());

    throw invalid_configuration_exception(name(), "Failed to open the path: " + file_path);
  }
}

void
print_string_process
::_step()
{
  edge_datum_t const input_dat = grab_from_port(priv::INPUT_PORT_NAME);

  switch (input_dat.get<0>()->type())
  {
    case datum::DATUM_DATA:
    {
      priv::string_t const input = input_dat.get<0>()->get_datum<priv::string_t>();

      d->fout << input << std::endl;
      break;
    }
    case datum::DATUM_EMPTY:
      break;
    case datum::DATUM_COMPLETE:
      mark_as_complete();
      break;
    case datum::DATUM_ERROR:
      break;
    case datum::DATUM_INVALID:
    default:
      break;
  }

  process::_step();
}

print_string_process::priv
::priv(path_t const& output_path)
  : path(output_path)
{
}

print_string_process::priv
::~priv()
{
}

}
