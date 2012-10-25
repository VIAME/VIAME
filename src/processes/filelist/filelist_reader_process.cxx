/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "filelist_reader_process.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/fstream.hpp>

#include <string>

/**
 * \file filelist_reader_process.cxx
 *
 * \brief Implementation of the filelist reader process.
 */

namespace vistk
{

class filelist_reader_process::priv
{
  public:
    priv(path_t const& input_path);
    ~priv();

    path_t const path;

    boost::filesystem::ifstream fin;

    static config::key_t const config_path;
    static port_t const port_output;
};

config::key_t const filelist_reader_process::priv::config_path = config::key_t("input");
process::port_t const filelist_reader_process::priv::port_output = port_t("path");

filelist_reader_process
::filelist_reader_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(
    priv::config_path,
    config::value_t(),
    config::description_t("The input file with a list of paths to read."));

  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(
    priv::port_output,
    port_type_t("path"),
    required,
    port_description_t("The paths from the file."));
}

filelist_reader_process
::~filelist_reader_process()
{
}

void
filelist_reader_process
::_configure()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path));
  }

  if (d->path.empty())
  {
    static std::string const reason = "The path given was empty";
    config::value_t const value = d->path.string<config::value_t>();

    throw invalid_configuration_value_exception(name(), priv::config_path, value, reason);
  }

  d->fin.open(d->path);

  if (!d->fin.good())
  {
    std::string const file_path = d->path.string<std::string>();
    std::string const reason = "Failed to open the path: " + file_path;

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
filelist_reader_process
::_step()
{
  datum_t dat;

  if (d->fin.eof())
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
  }
  else if (!d->fin.good())
  {
    static datum::error_t const err_string = datum::error_t("Error with input file stream.");

    dat = datum::error_datum(err_string);
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
      path_t const path = line;

      dat = datum::new_datum(path);
    }
  }

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

filelist_reader_process::priv
::priv(path_t const& input_path)
  : path(input_path)
{
}

filelist_reader_process::priv
::~priv()
{
}

}
