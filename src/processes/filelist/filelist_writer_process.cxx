/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "filelist_writer_process.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/fstream.hpp>

#include <string>

/**
 * \file filelist_writer_process.cxx
 *
 * \brief Implementation of the filelist writer process.
 */

namespace vistk
{

class filelist_writer_process::priv
{
  public:
    priv(path_t const& output_path);
    ~priv();

    path_t const path;

    boost::filesystem::ofstream fout;

    static config::key_t const config_path;
    static port_t const port_input;
};

config::key_t const filelist_writer_process::priv::config_path = config::key_t("output");
process::port_t const filelist_writer_process::priv::port_input = port_t("path");

filelist_writer_process
::filelist_writer_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(
    priv::config_path,
    config::value_t(),
    config::description_t("The file to write paths to."));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    port_type_t("path"),
    required,
    port_description_t("The paths that are to be written."));
}

filelist_writer_process
::~filelist_writer_process()
{
}

void
filelist_writer_process
::_configure()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path));
  }

  d->fout.open(d->path);

  if (!d->fout.good())
  {
    std::string const file_path = d->path.string<std::string>();
    std::string const reason = "Failed to open the path: " + file_path;

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
filelist_writer_process
::_reset()
{
  d->fout.close();

  process::_reset();
}

void
filelist_writer_process
::_step()
{
  path_t const path = grab_from_port_as<path_t>(priv::port_input);

  std::string const str = path.string<std::string>();

  d->fout << str << std::endl;

  process::_step();
}

filelist_writer_process::priv
::priv(path_t const& output_path)
  : path(output_path)
{
}

filelist_writer_process::priv
::~priv()
{
}

}
