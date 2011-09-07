/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "print_string_process.h"

#include <vistk/pipeline_types/basic_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/path.hpp>
#include <boost/make_shared.hpp>

#include <fstream>
#include <string>

/**
 * \file print_string_process.cxx
 *
 * \brief Implementation of the string printing process.
 */

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

    static config::key_t const config_path;
    static port_t const port_input;
};

config::key_t const print_string_process::priv::config_path = config::key_t("output");
process::port_t const print_string_process::priv::port_input = process::port_t("string");

print_string_process
::print_string_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The path of the file to output to.")));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    basic_types::t_string,
    required,
    port_description_t("Where strings are read from.")));
}

print_string_process
::~print_string_process()
{
}

void
print_string_process
::_init()
{
  // Configure the process.
  {
    priv::path_t const path = config_value<priv::path_t>(priv::config_path);

    d.reset(new priv(path));
  }

  boost::filesystem::path::string_type const path = d->path.native();

  if (path.empty())
  {
    config::value_t const value = config::value_t(path.begin(), path.end());

    throw invalid_configuration_value_exception(name(), priv::config_path, value, "The path given was empty");
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
  edge_datum_t const input_dat = grab_from_port(priv::port_input);

  switch (input_dat.get<0>()->type())
  {
    case datum::data:
    {
      priv::string_t const input = input_dat.get<0>()->get_datum<priv::string_t>();

      d->fout << input << std::endl;
      break;
    }
    case datum::complete:
      mark_as_complete();
      break;
    case datum::empty:
    case datum::error:
    case datum::invalid:
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
