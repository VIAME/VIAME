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
#include <vistk/pipeline/stamp.h>

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

    number_t current;

    bool has_color;

    stamp_t output_stamp;

    static number_t const DEFAULT_START_VALUE;
    static number_t const DEFAULT_END_VALUE;
    static config::key_t const START_CONFIG_NAME;
    static config::key_t const END_CONFIG_NAME;
    static port_t const OUTPUT_PORT_NAME;
    static port_t const COLOR_PORT_NAME;
};

number_process::priv::number_t const number_process::priv::DEFAULT_START_VALUE = 0;
number_process::priv::number_t const number_process::priv::DEFAULT_END_VALUE = 100;
config::key_t const number_process::priv::START_CONFIG_NAME = config::key_t("start");
config::key_t const number_process::priv::END_CONFIG_NAME = config::key_t("end");
process::port_t const number_process::priv::OUTPUT_PORT_NAME = process::port_t("number");
process::port_t const number_process::priv::COLOR_PORT_NAME = process::port_t("color");

number_process
::number_process(config_t const& config)
  : process(config)
{
  priv::number_t start = config->get_value<priv::number_t>(priv::START_CONFIG_NAME, priv::DEFAULT_START_VALUE);
  priv::number_t end = config->get_value<priv::number_t>(priv::END_CONFIG_NAME, priv::DEFAULT_END_VALUE);

  d = boost::shared_ptr<priv>(new priv(start, end));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::COLOR_PORT_NAME, port_info_t(new port_info(
    type_none,
    port_flags_t(),
    port_description_t("If connected, uses the stamp's color for the output."))));
  declare_output_port(priv::OUTPUT_PORT_NAME, port_info_t(new port_info(
    port_types::t_integer,
    required,
    port_description_t("Where the numbers will be available."))));

  declare_configuration_key(priv::START_CONFIG_NAME, conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::DEFAULT_START_VALUE),
    config::description_t("The value to start counting at."))));
  declare_configuration_key(priv::END_CONFIG_NAME, conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::DEFAULT_END_VALUE),
    config::description_t("The value to stop counting at."))));
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

  if (!input_port_edge(priv::COLOR_PORT_NAME).expired())
  {
    d->has_color = true;
  }

  d->output_stamp = heartbeat_stamp();
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

  d->output_stamp = stamp::incremented_stamp(d->output_stamp);

  if (d->has_color)
  {
    edge_datum_t const color_dat = grab_from_port(priv::COLOR_PORT_NAME);

    switch (color_dat.get<0>()->type())
    {
      case datum::DATUM_COMPLETE:
        mark_as_complete();
        dat = datum::complete_datum();
      case datum::DATUM_DATA:
      case datum::DATUM_EMPTY:
        d->output_stamp = stamp::recolored_stamp(d->output_stamp, color_dat.get<1>());
        break;
      case datum::DATUM_ERROR:
        dat = datum::error_datum("Error on the color input edge.");
        break;
      case datum::DATUM_INVALID:
      default:
        dat = datum::error_datum("Unrecognized datum type.");
        break;
    }

  }

  edge_datum_t const edat = edge_datum_t(dat, d->output_stamp);

  push_to_port(priv::OUTPUT_PORT_NAME, edat);

  process::_step();
}

number_process::priv
::priv(number_t s, number_t e)
  : start(s)
  , end(e)
  , current(s)
  , has_color(false)
{
}

number_process::priv
::~priv()
{
}

}
