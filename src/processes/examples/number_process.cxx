/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "number_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/foreach.hpp>

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

    edges_t output_edges;

    static port_t const output_port_name;
};

process::port_t const number_process::priv::output_port_name = process::port_t("number");

number_process
::number_process(config_t const& config)
  : process(config)
{
  priv::number_t start = config->get_value<priv::number_t>("start", 0);
  priv::number_t end = config->get_value<priv::number_t>("end", 100);

  d = boost::shared_ptr<priv>(new priv(start, end));
}

number_process
::~number_process()
{
}

process_registry::type_t
number_process
::type() const
{
  return process_registry::type_t("number_process");
}

void
number_process
::_init()
{
  // Check the configuration.
  if (d->end <= d->start)
  {
    /// \todo Throw an exception for a misconfiguration.
  }

  // Ensure the output port is connected.
  if (!d->output_edges.size())
  {
    /// \todo Throw an exception due to a port not being connected.
  }
}

void
number_process
::_step()
{
  datum_t dat;

  if (d->current == d->end)
  {
    dat = datum::complete_datum();
  }
  else
  {
    dat = datum::new_datum(d->current);

    ++d->current;
  }

  BOOST_FOREACH (edge_t& edge, d->output_edges)
  {
    edge->push_datum(edge_datum_t(dat, heartbeat_stamp()));
  }

  process::_step();
}

void
number_process
::_connect_output_port(port_t const& port, edge_t edge)
{
  if (port == d->output_port_name)
  {
    d->output_edges.push_back(edge);
  }

  process::_connect_output_port(port, edge);
}

process::ports_t
number_process
::_output_ports() const
{
  ports_t ports;

  ports.push_back(d->output_port_name);

  return ports;
}

number_process::priv
::priv(number_t s, number_t e)
  : start(s)
  , end(e)
  , current(s)
{
}

number_process::priv
::~priv()
{
}

} // end namespace vistk
