/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "print_number_process.h"

#include <vistk/pipeline_types/port_types.h>

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/filesystem/path.hpp>

#include <iostream>
#include <fstream>

namespace vistk
{

class print_number_process::priv
{
  public:
    typedef uint32_t number_t;
    typedef boost::filesystem::path path_t;

    priv(path_t const& output_path);
    ~priv();

    path_t const path;

    edge_t input_edge;

    std::ofstream fout;

    static config::key_t const CONFIG_OUTPUT_NAME;
    static port_t const INPUT_PORT_NAME;
};

config::key_t const print_number_process::priv::CONFIG_OUTPUT_NAME = config::key_t("output");
process::port_t const print_number_process::priv::INPUT_PORT_NAME = process::port_t("number");

print_number_process
::print_number_process(config_t const& config)
  : process(config)
{
  priv::path_t path = config->get_value<priv::path_t>(priv::CONFIG_OUTPUT_NAME, priv::path_t());

  d = boost::shared_ptr<priv>(new priv(path));
}

print_number_process
::~print_number_process()
{
}

process_registry::type_t
print_number_process
::type() const
{
  return process_registry::type_t("print_number_process");
}

void
print_number_process
::_init()
{
  d->fout.open(d->path.native().c_str());

  if (!d->fout.good())
  {
    /// \todo Throw exception.
  }
}

void
print_number_process
::_step()
{
  edge_datum_t const input_dat = d->input_edge->get_datum();

  switch (input_dat.get<0>()->type())
  {
    case datum::DATUM_DATA:
    {
      priv::number_t const input = input_dat.get<0>()->get_datum<priv::number_t>();

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

void
print_number_process
::_connect_input_port(port_t const& port, edge_t edge)
{
  if (port == priv::INPUT_PORT_NAME)
  {
    if (d->input_edge)
    {
      throw port_reconnect_exception(name(), port);
    }

    d->input_edge = edge;

    return;
  }

  process::_connect_input_port(port, edge);
}

process::port_type_t
print_number_process
::_input_port_type(port_t const& port) const
{
  if (port == priv::INPUT_PORT_NAME)
  {
    port_flags_t flags;

    flags.insert(flag_required);

    return port_type_t(port_types::t_integer, flags);
  }

  return process::_input_port_type(port);
}

process::port_description_t
print_number_process
::_input_port_description(port_t const& port) const
{
  if (port == priv::INPUT_PORT_NAME)
  {
    return port_description_t("Where numbers are read from.");
  }

  return process::_input_port_description(port);
}

process::ports_t
print_number_process
::_input_ports() const
{
  ports_t ports;

  ports.push_back(priv::INPUT_PORT_NAME);

  return ports;
}

config::keys_t
print_number_process
::_available_config() const
{
  config::keys_t keys = process::_available_config();

  keys.push_back(priv::CONFIG_OUTPUT_NAME);

  return keys;
}

config::value_t
print_number_process
::_config_default(config::key_t const& key) const
{
  if (key == priv::CONFIG_OUTPUT_NAME)
  {
    return config::value_t();
  }

  return process::_config_default(key);
}

config::description_t
print_number_process
::_config_description(config::key_t const& key) const
{
  if (key == priv::CONFIG_OUTPUT_NAME)
  {
    return config::description_t("The path the file to output to");
  }

  return process::_config_description(key);
}

print_number_process::priv
::priv(path_t const& output_path)
  : path(output_path)
{
}

print_number_process::priv
::~priv()
{
}

}
