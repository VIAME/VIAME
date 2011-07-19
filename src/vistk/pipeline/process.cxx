/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "process.h"
#include "process_exception.h"

#include "config.h"
#include "datum.h"
#include "edge.h"
#include "stamp.h"
#include "types.h"

#include <boost/algorithm/string/predicate.hpp>

#include <utility>

/**
 * \file process.cxx
 *
 * \brief Implementation of the base class for \link process processes\endlink.
 */

namespace vistk
{

class process::priv
{
  public:
    priv();
    ~priv();

    void run_heartbeat();

    name_t name;

    typedef std::pair<edge_t, edge_t> edge_pair_t;
    typedef std::map<port_t, edge_pair_t> edge_map_t;

    edges_t heartbeats;

    edges_t input_edges;
    edges_t output_edges;

    bool is_complete;

    stamp_t hb_stamp;

    static port_t const HEARTBEAT_PORT_NAME;
    static config::key_t const NAME_CONFIG_KEY;
    static config::value_t const DEFAULT_PROCESS_NAME;
};

process::port_t const process::priv::HEARTBEAT_PORT_NAME = "heartbeat";
config::key_t const process::priv::NAME_CONFIG_KEY = "_name";
config::key_t const process::priv::DEFAULT_PROCESS_NAME = "(unnamed)";

void
process
::init()
{
}

void
process
::step()
{
  /// \todo Pre-step

  if (d->is_complete)
  {
    /// \todo Determine timestamp to use
  }
  else
  {
    _step();
  }

  /// \todo Post-step

  d->run_heartbeat();
}

void
process
::connect_input_port(port_t const& port, edge_t edge)
{
  _connect_input_port(port, edge);
}

void
process
::connect_output_port(port_t const& port, edge_t edge)
{
  if (!edge)
  {
    throw null_edge_port_connection(d->name, port);
  }

  if (port == priv::HEARTBEAT_PORT_NAME)
  {
    d->heartbeats.push_back(edge);

    return;
  }

  _connect_output_port(port, edge);
}

process::ports_t
process
::input_ports() const
{
  ports_t ports = _input_ports();

  return ports;
}

process::ports_t
process
::output_ports() const
{
  ports_t ports = _output_ports();

  ports.push_back(d->HEARTBEAT_PORT_NAME);

  return ports;
}

config::value_t
process
::config_default(config::key_t const& key) const
{
  if (key == priv::NAME_CONFIG_KEY)
  {
    return boost::lexical_cast<config::value_t>(priv::DEFAULT_PROCESS_NAME);
  }

  throw unknown_configuration_value(d->name, key);
}

config::description_t
process
::config_description(config::key_t const& key) const
{
  if (key == priv::NAME_CONFIG_KEY)
  {
    return config::description_t("The name of the process");
  }

  throw unknown_configuration_value(d->name, key);
}

process::name_t
process
::name() const
{
  return d->name;
}

process
::process(config_t const& config)
  : d(new priv)
{
  d->name = config->get_value<name_t>(priv::NAME_CONFIG_KEY, priv::DEFAULT_PROCESS_NAME);
}

process
::~process()
{
}

void
process
::_init()
{
}

void
process
::_step()
{
}

void
process
::_connect_input_port(port_t const& port, edge_t /*edge*/)
{
  throw no_such_port_exception(d->name, port);
}

void
process
::_connect_output_port(port_t const& port, edge_t /*edge*/)
{
  throw no_such_port_exception(d->name, port);
}

process::ports_t
process
::_input_ports() const
{
  return ports_t();
}

process::ports_t
process
::_output_ports() const
{
  return ports_t();
}

void
process
::mark_as_complete()
{
  d->is_complete = true;
}

stamp_t
process
::heartbeat_stamp() const
{
  return d->hb_stamp;
}

bool
process
::same_colored_edges(edges_t const& edges)
{
  edges_t::const_iterator it = edges.begin();
  edges_t::const_iterator it_end = edges.end();

  for ( ; it != it_end; ++it)
  {
    edges_t::const_iterator it2 = it;

    stamp_t const st = (*it)->peek_datum().get<1>();

    for (++it2; it2 != it_end; ++it2)
    {
      if (!st->is_same_color((*it2)->peek_datum().get<1>()))
      {
        return false;
      }
    }
  }

  return true;
}

bool
process
::sync_edges(edges_t const& edges)
{
  stamp_t max_stamp;

  edges_t::const_iterator it = edges.begin();
  edges_t::const_iterator it_end = edges.end();

  /// \todo Compute whether the given edges are synchronized.
}

process::priv
::priv()
  : is_complete(false)
  , hb_stamp(stamp::new_stamp())
{
}

process::priv
::~priv()
{
}

void
process::priv
::run_heartbeat()
{
  datum_t dat;

  if (is_complete)
  {
    dat = datum::complete_datum();
  }
  else
  {
    dat = datum::empty_datum();
  }

  edge_datum_t edge_dat(dat, hb_stamp);

  hb_stamp = stamp::incremented_stamp(hb_stamp);

  edges_t::iterator hb = heartbeats.begin();
  edges_t::iterator hb_end = heartbeats.end();

  for ( ; hb != hb_end; ++hb)
  {
    (*hb)->push_datum(edge_dat);
  }
}

} // end namespace vistk
