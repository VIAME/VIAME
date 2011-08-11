/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "number_process.h"

#include <vistk/pipeline_types/port_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

namespace vistk
{

class number_process::priv
{
  public:
    typedef uint32_t number_t;

    priv(number_t s, number_t e);
    ~priv();

    number_t const start;
    number_t const end;

    conf_info_t start_conf_info;
    conf_info_t end_conf_info;

    edge_group_t output_edges;

    port_info_t output_port_info;

    number_t current;

    static number_t const DEFAULT_START_VALUE;
    static number_t const DEFAULT_END_VALUE;
    static config::key_t const START_CONFIG_NAME;
    static config::key_t const END_CONFIG_NAME;
    static port_t const OUTPUT_PORT_NAME;
};

number_process::priv::number_t const number_process::priv::DEFAULT_START_VALUE = 0;
number_process::priv::number_t const number_process::priv::DEFAULT_END_VALUE = 100;
config::key_t const number_process::priv::START_CONFIG_NAME = config::key_t("start");
config::key_t const number_process::priv::END_CONFIG_NAME = config::key_t("end");
process::port_t const number_process::priv::OUTPUT_PORT_NAME = process::port_t("number");

number_process
::number_process(config_t const& config)
  : process(config)
{
  priv::number_t start = config->get_value<priv::number_t>(priv::START_CONFIG_NAME, priv::DEFAULT_START_VALUE);
  priv::number_t end = config->get_value<priv::number_t>(priv::END_CONFIG_NAME, priv::DEFAULT_END_VALUE);

  d = boost::shared_ptr<priv>(new priv(start, end));
}

number_process
::~number_process()
{
}

void
number_process
::_init()
{
  // Check the configuration.
  if (d->end <= d->start)
  {
    throw invalid_configuration_exception(name(), "The start value must be greater than the end value");
  }
}

void
number_process
::_step()
{
  datum_t dat;

  if (d->current == d->end)
  {
    mark_as_complete();
    dat = datum::complete_datum();
  }
  else
  {
    dat = datum::new_datum(d->current);

    ++d->current;
  }

  edge_datum_t const edat = edge_datum_t(dat, heartbeat_stamp());

  push_to_edges(d->output_edges, edat);

  process::_step();
}

void
number_process
::_connect_output_port(port_t const& port, edge_ref_t edge)
{
  if (port == priv::OUTPUT_PORT_NAME)
  {
    d->output_edges.push_back(edge);

    return;
  }

  process::_connect_output_port(port, edge);
}

process::port_info_t
number_process
::_output_port_info(port_t const& port) const
{
  if (port == priv::OUTPUT_PORT_NAME)
  {
    return d->output_port_info;
  }

  return process::_output_port_info(port);
}

process::ports_t
number_process
::_output_ports() const
{
  ports_t ports;

  ports.push_back(priv::OUTPUT_PORT_NAME);

  return ports;
}

config::keys_t
number_process
::_available_config() const
{
  config::keys_t keys = process::_available_config();

  keys.push_back(priv::START_CONFIG_NAME);
  keys.push_back(priv::END_CONFIG_NAME);

  return keys;
}

process::conf_info_t
number_process
::_config_info(config::key_t const& key) const
{
  if (key == priv::START_CONFIG_NAME)
  {
    return d->start_conf_info;
  }
  if (key == priv::END_CONFIG_NAME)
  {
    return d->end_conf_info;
  }

  return process::_config_info(key);
}

number_process::priv
::priv(number_t s, number_t e)
  : start(s)
  , end(e)
  , current(s)
{
  port_flags_t required;

  required.insert(flag_required);

  output_port_info = port_info_t(new port_info(
    port_types::t_integer,
    required,
    port_description_t("Where the numbers will be available.")));

  start_conf_info = conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::DEFAULT_START_VALUE),
    config::description_t("The value to start counting at.")));
  end_conf_info = conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::DEFAULT_END_VALUE),
    config::description_t("The value to stop counting at.")));
}

number_process::priv
::~priv()
{
}

}
