/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "number_process.h"

#include <vistk/pipeline_types/basic_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/make_shared.hpp>

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

    static number_t const default_start;
    static number_t const default_end;
    static config::key_t const config_start;
    static config::key_t const config_end;
    static port_t const port_output;
    static port_t const port_color;
};

number_process::priv::number_t const number_process::priv::default_start = 0;
number_process::priv::number_t const number_process::priv::default_end = 100;
config::key_t const number_process::priv::config_start = config::key_t("start");
config::key_t const number_process::priv::config_end = config::key_t("end");
process::port_t const number_process::priv::port_output = process::port_t("number");
process::port_t const number_process::priv::port_color = process::port_t("color");

number_process
::number_process(config_t const& config)
  : process(config)
{
  priv::number_t start = config->get_value<priv::number_t>(priv::config_start, priv::default_start);
  priv::number_t end = config->get_value<priv::number_t>(priv::config_end, priv::default_end);

  d = boost::make_shared<priv>(start, end);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_color, boost::make_shared<port_info>(
    type_none,
    port_flags_t(),
    port_description_t("If connected, uses the stamp's color for the output.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    basic_types::t_integer,
    required,
    port_description_t("Where the numbers will be available.")));

  declare_configuration_key(priv::config_start, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_start),
    config::description_t("The value to start counting at.")));
  declare_configuration_key(priv::config_end, boost::make_shared<conf_info>(
    boost::lexical_cast<config::value_t>(priv::default_end),
    config::description_t("The value to stop counting at.")));
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

  if (!input_port_edge(priv::port_color).expired())
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
