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
#include <boost/foreach.hpp>

#include <utility>

/**
 * \file process.cxx
 *
 * \brief Implementation of the base class for \link vistk::process processes\endlink.
 */

namespace vistk
{

process::port_t const process::port_heartbeat = port_t("heartbeat");
config::key_t const process::config_name = config::key_t("_name");
process::port_type_name_t const process::type_any = port_type_name_t("_any");
process::port_type_name_t const process::type_none = port_type_name_t("_none");
process::port_flag_t const process::flag_output_const = port_flag_t("_const");
process::port_flag_t const process::flag_input_mutable = port_flag_t("_mutable");
process::port_flag_t const process::flag_required = port_flag_t("_required");

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

    static config::value_t const DEFAULT_PROCESS_NAME;
};

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
  /// \todo Make reentrant.

  /// \todo Are there any pre-_step actions?

  if (d->is_complete)
  {
    /// \todo What exactly should be done here?
  }
  else
  {
    _step();
  }

  /// \todo Are there any post-_step actions?

  d->run_heartbeat();
}

bool
process
::is_reentrant() const
{
  return false;
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
    throw null_edge_port_connection_exception(d->name, port);
  }

  if (port == port_heartbeat)
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

  ports.push_back(port_heartbeat);

  return ports;
}

process::port_type_t
process
::input_port_type(port_t const& port) const
{
  return _input_port_type(port);
}

process::port_type_t
process
::output_port_type(port_t const& port) const
{
  if (port == port_heartbeat)
  {
    return port_type_t(type_none, port_flags_t());
  }

  return _output_port_type(port);
}

process::port_description_t
process
::input_port_description(port_t const& port) const
{
  return _input_port_description(port);
}

process::port_description_t
process
::output_port_description(port_t const& port) const
{
  if (port == port_heartbeat)
  {
    return port_description_t("Outputs the hearbeat stamp with an empty datum.");
  }

  return _output_port_description(port);
}

config::keys_t
process
::available_config() const
{
  return config::keys_t();
}

config::value_t
process
::config_default(config::key_t const& key) const
{
  if (key == config_name)
  {
    return boost::lexical_cast<config::value_t>(priv::DEFAULT_PROCESS_NAME);
  }

  throw unknown_configuration_value_exception(d->name, key);
}

config::description_t
process
::config_description(config::key_t const& key) const
{
  if (key == config_name)
  {
    return config::description_t("The name of the process");
  }

  throw unknown_configuration_value_exception(d->name, key);
}

process::name_t
process
::name() const
{
  return d->name;
}

process
::process(config_t const& config) throw()
{
  d = boost::shared_ptr<priv>(new priv);

  d->name = config->get_value<name_t>(config_name, priv::DEFAULT_PROCESS_NAME);
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

process::port_type_t
process
::_input_port_type(port_t const& port) const
{
  throw no_such_port_exception(d->name, port);
}

process::port_type_t
process
::_output_port_type(port_t const& port) const
{
  throw no_such_port_exception(d->name, port);
}

process::port_description_t
process
::_input_port_description(port_t const& port) const
{
  throw no_such_port_exception(d->name, port);
}

process::port_description_t
process
::_output_port_description(port_t const& port) const
{
  throw no_such_port_exception(d->name, port);
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
::syncd_edges(edges_t const& edges)
{
  edges_t::const_iterator it = edges.begin();
  edges_t::const_iterator it_end = edges.end();

  for ( ; it != it_end; ++it)
  {
    edges_t::const_iterator it2 = it;

    stamp_t const st = (*it)->peek_datum().get<1>();

    for (++it2; it2 != it_end; ++it2)
    {
      stamp_t const st2 = (*it2)->peek_datum().get<1>();

      if (*st != *st2)
      {
        return false;
      }
    }
  }

  return true;
}

void
process
::push_to_edges(edges_t const& edges, edge_datum_t const& dat)
{
  BOOST_FOREACH (edge_t e, edges)
  {
    e->push_datum(dat);
  }
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
