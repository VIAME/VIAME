/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "timestamp_reader_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <vistk/utilities/path.h>
#include <vistk/utilities/timestamp.h>

#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>

#include <fstream>
#include <string>

/**
 * \file timestamp_reader_process.cxx
 *
 * \brief Implementation of the timestamp reader process.
 */

namespace vistk
{

class timestamp_reader_process::priv
{
  public:
    priv(path_t const& input_path);
    ~priv();

    path_t const path;

    std::ifstream fin;

    static config::key_t const config_path;
    static port_t const port_output;
};

config::key_t const timestamp_reader_process::priv::config_path = config::key_t("path");
process::port_t const timestamp_reader_process::priv::port_output = port_t("timestamp");

timestamp_reader_process
::timestamp_reader_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_path, boost::make_shared<conf_info>(
    config::value_t(),
    config::description_t("The path to the file to read")));

  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    "timestamp",
    required,
    port_description_t("Where the timestamps will be available.")));
}

timestamp_reader_process
::~timestamp_reader_process()
{
}

void
timestamp_reader_process
::_init()
{
  // Configure the process.
  {
    path_t const path = config_value<path_t>(priv::config_path);

    d.reset(new priv(path));
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

  process::_init();
}

void
timestamp_reader_process
::_step()
{
  datum_t dat;
  bool read_error = false;
  bool complete = false;

  if (d->fin.eof())
  {
    complete = true;
  }
  else if (!d->fin.good())
  {
    static datum::error_t const err_string = datum::error_t("Error with input file stream.");

    read_error = true;

    dat = datum::error_datum(err_string);
  }
  else
  {
    std::string time_str;
    std::string frame_str;

    if (!(d->fin >> time_str) ||
        !(d->fin >> frame_str))
    {
      read_error = true;
    }

    if (d->fin.eof())
    {
      complete = true;
    }

    if (!read_error && !complete)
    {
      static std::string const undefined_string = "-";

      boost::optional<timestamp::time_t> time;
      boost::optional<timestamp::frame_t> frame;

      if (time_str != undefined_string)
      {
        time = boost::lexical_cast<timestamp::time_t>(time_str);
      }

      if (frame_str != undefined_string)
      {
        frame = boost::lexical_cast<timestamp::frame_t>(frame_str);
      }

      if (time && frame)
      {
        dat = datum::new_datum(timestamp(*time, *frame));
      }
      else if (time)
      {
        dat = datum::new_datum(timestamp(*time));
      }
      else if (frame)
      {
        dat = datum::new_datum(timestamp(*frame));
      }
      else
      {
        dat = datum::new_datum(timestamp());
      }
    }
  }

  if (read_error)
  {
    static datum::error_t const err_string = datum::error_t("Error reading from the input file.");

    dat = datum::error_datum(err_string);
  }

  if (complete)
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
  }

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

timestamp_reader_process::priv
::priv(path_t const& input_path)
  : path(input_path)
{
}

timestamp_reader_process::priv
::~priv()
{
}

}
