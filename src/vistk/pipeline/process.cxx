/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <map>
#include <utility>

/**
 * \file process.cxx
 *
 * \brief Implementation of the base class for \link vistk::process processes\endlink.
 */

namespace vistk
{

process::constraint_t const process::constraint_no_threads = constraint_t("_no_thread");
process::constraint_t const process::constraint_python = constraint_t("_python");
process::constraint_t const process::constraint_no_reentrancy = constraint_t("_no_reentrant");
process::constraint_t const process::constraint_unsync_input = constraint_t("_unsync_input");
process::constraint_t const process::constraint_unsync_output = constraint_t("_unsync_output");
process::port_t const process::port_heartbeat = port_t("_heartbeat");
config::key_t const process::config_name = config::key_t("_name");
config::key_t const process::config_type = config::key_t("_type");
process::port_type_t const process::type_any = port_type_t("_any");
process::port_type_t const process::type_none = port_type_t("_none");
process::port_type_t const process::type_data_dependent = port_type_t("_data_dependent");
process::port_type_t const process::type_flow_dependent = port_type_t("_flow_dependent/");
process::port_flag_t const process::flag_output_const = port_flag_t("_const");
process::port_flag_t const process::flag_input_static = port_flag_t("_static");
process::port_flag_t const process::flag_input_mutable = port_flag_t("_mutable");
process::port_flag_t const process::flag_input_nodep = port_flag_t("_nodep");
process::port_flag_t const process::flag_required = port_flag_t("_required");
config::key_t const process::static_input_prefix = config::key_t("static/");

process::port_info
::port_info(port_type_t const& type_,
            port_flags_t const& flags_,
            port_description_t const& description_,
            port_frequency_t const& frequency_)
  : type(type_)
  , flags(flags_)
  , description(description_)
  , frequency(frequency_)
{
}

process::port_info
::~port_info()
{
}

process::conf_info
::conf_info(config::value_t const& def_,
            config::description_t const& description_)
  : def(def_)
  , description(description_)
{
}

process::conf_info
::~conf_info()
{
}

process::data_info
::data_info(bool in_sync_,
            datum::type_t max_status_)
  : in_sync(in_sync_)
  , max_status(max_status_)
{
}

process::data_info
::~data_info()
{
}

class process::priv
{
  public:
    priv(process* proc, config_t const& c);
    ~priv();

    void run_heartbeat();
    bool connect_input_port(port_t const& port, edge_t const& edge);
    bool connect_output_port(port_t const& port, edge_t const& edge);

    datum_t check_required_input();
    void grab_from_input_edges();
    void push_to_output_edges(datum_t const& dat) const;
    bool required_outputs_done() const;

    name_t name;
    type_t type;

    typedef std::map<port_t, port_info_t> port_map_t;
    typedef std::map<config::key_t, conf_info_t> conf_map_t;

    typedef boost::shared_mutex mutex_t;
    typedef boost::shared_lock<mutex_t> shared_lock_t;
    typedef boost::upgrade_lock<mutex_t> upgrade_lock_t;
    typedef boost::unique_lock<mutex_t> unique_lock_t;
    typedef boost::upgrade_to_unique_lock<mutex_t> upgrade_to_unique_lock_t;

    typedef boost::tuple<mutex_t, edges_t, stamp_t> output_port_info;
    typedef boost::shared_ptr<output_port_info> output_port_info_t;

    typedef std::map<port_t, edge_t> input_edge_map_t;
    typedef std::map<port_t, output_port_info_t> output_edge_map_t;

    typedef port_t tag_t;

    typedef boost::optional<port_type_t> flow_tag_port_type_t;
    typedef std::map<tag_t, flow_tag_port_type_t> flow_tag_port_type_map_t;

    typedef std::map<tag_t, ports_t> flow_tag_port_map_t;

    typedef std::map<port_t, tag_t> port_tag_map_t;

    typedef boost::optional<port_frequency_t> core_frequency_t;

    tag_t port_flow_tag_name(port_type_t const& port_type) const;
    void check_tag(tag_t const& tag);

    port_map_t input_ports;
    port_map_t output_ports;

    conf_map_t config_keys;

    input_edge_map_t input_edges;
    output_edge_map_t output_edges;

    process* const q;
    config_t const conf;

    ports_t static_inputs;
    ports_t required_inputs;
    ports_t required_outputs;

    flow_tag_port_type_map_t flow_tag_port_types;
    flow_tag_port_map_t input_flow_tag_ports;
    flow_tag_port_map_t output_flow_tag_ports;

    port_tag_map_t input_port_tags;
    port_tag_map_t output_port_tags;

    core_frequency_t core_frequency;

    bool configured;
    bool initialized;
    bool is_complete;

    bool input_sync;
    bool input_valid;

    stamp_t stamp_for_inputs;

    static config::value_t const default_name;
};

config::value_t const process::priv::default_name = "(unnamed)";

void
process
::configure()
{
  if (d->initialized)
  {
    throw already_initialized_exception(d->name);
  }

  if (d->configured)
  {
    throw reconfigured_exception(d->name);
  }

  _configure();

  d->configured = true;
}

void
process
::init()
{
  if (!d->configured)
  {
    throw unconfigured_exception(d->name);
  }

  if (d->initialized)
  {
    throw reinitialization_exception(d->name);
  }

  _init();

  d->initialized = true;
}

void
process
::reset()
{
  _reset();
}

void
process
::step()
{
  if (!d->configured)
  {
    throw unconfigured_exception(d->name);
  }

  if (!d->core_frequency)
  {
    throw uninitialized_exception(d->name);
  }

  /// \todo Make reentrant.

  /// \todo Are there any pre-_step actions?

  if (d->is_complete)
  {
    /// \todo What exactly should be done here?
  }
  else
  {
    datum_t const dat = d->check_required_input();

    if (dat)
    {
      d->grab_from_input_edges();
      d->push_to_output_edges(dat);
    }
    else
    {
      _step();
    }

    d->stamp_for_inputs = stamp_t();
  }

  /// \todo Are there any post-_step actions?

  d->run_heartbeat();

  /// \todo Should this really be done here?
  if (d->required_outputs_done())
  {
    mark_process_as_complete();
  }
}

process::constraints_t
process
::constraints() const
{
  return _constraints();
}

void
process
::connect_input_port(port_t const& port, edge_t edge)
{
  if (!edge)
  {
    throw null_edge_port_connection_exception(d->name, port);
  }

  if (d->initialized)
  {
    throw connect_to_initialized_process_exception(d->name, port);
  }

  if (!d->connect_input_port(port, edge))
  {
    _connect_input_port(port, edge);
  }
}

void
process
::connect_output_port(port_t const& port, edge_t edge)
{
  if (!edge)
  {
    throw null_edge_port_connection_exception(d->name, port);
  }

  if (!d->connect_output_port(port, edge))
  {
    _connect_output_port(port, edge);
  }
}

process::ports_t
process
::input_ports() const
{
  ports_t ports = _input_ports();

  BOOST_FOREACH (priv::port_map_t::value_type const& port, d->input_ports)
  {
    port_t const& port_name = port.first;

    ports.push_back(port_name);
  }

  return ports;
}

process::ports_t
process
::output_ports() const
{
  ports_t ports = _output_ports();

  BOOST_FOREACH (priv::port_map_t::value_type const& port, d->output_ports)
  {
    port_t const& port_name = port.first;

    ports.push_back(port_name);
  }

  return ports;
}

process::port_info_t
process
::input_port_info(port_t const& port)
{
  return _input_port_info(port);
}

process::port_info_t
process
::output_port_info(port_t const& port)
{
  return _output_port_info(port);
}

bool
process
::set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  if (d->initialized)
  {
    throw set_type_on_initialized_process_exception(d->name, port, new_type);
  }

  return _set_input_port_type(port, new_type);
}

bool
process
::set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  if (d->initialized)
  {
    throw set_type_on_initialized_process_exception(d->name, port, new_type);
  }

  return _set_output_port_type(port, new_type);
}

config::keys_t
process
::available_config() const
{
  config::keys_t keys = _available_config();

  BOOST_FOREACH (priv::conf_map_t::value_type const& conf, d->config_keys)
  {
    config::key_t const& key = conf.first;

    keys.push_back(key);
  }

  return keys;
}

process::conf_info_t
process
::config_info(config::key_t const& key)
{
  return _config_info(key);
}

process::name_t
process
::name() const
{
  return d->name;
}

process::type_t
process
::type() const
{
  return d->type;
}

process
::process(config_t const& config)
{
  if (!config)
  {
    throw null_process_config_exception();
  }

  d.reset(new priv(this, config));

  declare_configuration_key(
    config_name,
    config::value_t(),
    config::description_t("The name of the process."));
  declare_configuration_key(
    config_type,
    config::value_t(),
    config::description_t("The type of the process."));

  d->name = config_value<name_t>(config_name);
  d->type = config_value<type_t>(config_type);

  declare_output_port(
    port_heartbeat,
    type_none,
    port_flags_t(),
    port_description_t("Outputs the heartbeat stamp with an empty datum."),
    port_frequency_t(1));
}

process
::~process()
{
}

void
process
::_configure()
{
}

void
process
::_init()
{
  BOOST_FOREACH (priv::output_edge_map_t::value_type& oport, d->output_edges)
  {
    port_t const& port_name = oport.first;

    port_info_t const info = output_port_info(port_name);
    port_frequency_t const& port_frequency = info->frequency;

    // Skip ports with an unknown port frequency.
    if (!port_frequency)
    {
      continue;
    }

    port_frequency_t const port_run_frequency = (*d->core_frequency) * port_frequency;

    if (port_run_frequency.denominator() != 1)
    {
      static std::string const reason = "A port has a runtime frequency "
                                        "that is not a whole number";

      throw std::runtime_error(reason);
    }

    stamp::increment_t const port_increment = port_run_frequency.numerator();

    {
      priv::output_port_info_t& oinfo = oport.second;
      priv::mutex_t& mut = oinfo->get<0>();

      priv::unique_lock_t const lock(mut);

      (void)lock;

      stamp_t& stamp = oinfo->get<2>();

      stamp = stamp::new_stamp(port_increment);
    }
  }
}

void
process
::_reset()
{
  d->input_edges.clear();

  BOOST_FOREACH (priv::output_edge_map_t::value_type& oport, d->output_edges)
  {
    priv::output_port_info_t& info = oport.second;
    priv::mutex_t& mut = info->get<0>();

    priv::unique_lock_t const lock(mut);

    (void)lock;

    edges_t& edges = info->get<1>();
    stamp_t& stamp = info->get<2>();

    edges.clear();
    stamp.reset();
  }

  d->configured = false;
  d->initialized = false;
  d->core_frequency.reset();
}

void
process
::_step()
{
}

process::constraints_t
process
::_constraints() const
{
  constraints_t consts;

  consts.insert(constraint_no_reentrancy);

  return consts;
}

void
process
::_connect_input_port(port_t const& port, edge_t edge)
{
  if (!d->connect_input_port(port, edge))
  {
    throw no_such_port_exception(d->name, port);
  }
}

void
process
::_connect_output_port(port_t const& port, edge_t edge)
{
  if (!d->connect_output_port(port, edge))
  {
    throw no_such_port_exception(d->name, port);
  }
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

process::port_info_t
process
::_input_port_info(port_t const& port)
{
  priv::port_map_t::const_iterator const i = d->input_ports.find(port);

  if (i != d->input_ports.end())
  {
    return i->second;
  }

  throw no_such_port_exception(d->name, port);
}

process::port_info_t
process
::_output_port_info(port_t const& port)
{
  priv::port_map_t::const_iterator const i = d->output_ports.find(port);

  if (i != d->output_ports.end())
  {
    return i->second;
  }

  throw no_such_port_exception(d->name, port);
}

bool
process
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  port_info_t const info = input_port_info(port);
  port_type_t const& old_type = info->type;

  if (old_type == new_type)
  {
    return true;
  }

  bool const is_flow_dependent = boost::starts_with(old_type, type_flow_dependent);

  if ((old_type != type_data_dependent) && !is_flow_dependent)
  {
    throw static_type_reset_exception(name(), port, old_type, new_type);
  }

  if (is_flow_dependent)
  {
    priv::tag_t const tag = d->port_flow_tag_name(old_type);

    if (!tag.empty())
    {
      ports_t const& iports = d->input_flow_tag_ports[tag];

      BOOST_FOREACH (port_t const& iport, iports)
      {
        declare_input_port(
          iport,
          new_type,
          info->flags,
          info->description,
          info->frequency);
      }

      ports_t const& oports = d->output_flow_tag_ports[tag];

      BOOST_FOREACH (port_t const& oport, oports)
      {
        declare_output_port(
          oport,
          new_type,
          info->flags,
          info->description,
          info->frequency);
      }

      d->flow_tag_port_types[tag] = new_type;

      return true;
    }
  }

  declare_input_port(
    port,
    new_type,
    info->flags,
    info->description,
    info->frequency);

  return true;
}

bool
process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  port_info_t const info = output_port_info(port);
  port_type_t const& old_type = info->type;

  if (old_type == new_type)
  {
    return true;
  }

  bool const is_flow_dependent = boost::starts_with(old_type, type_flow_dependent);

  if ((old_type != type_data_dependent) && !is_flow_dependent)
  {
    throw static_type_reset_exception(name(), port, old_type, new_type);
  }

  if (is_flow_dependent)
  {
    priv::tag_t const tag = d->port_flow_tag_name(old_type);

    if (!tag.empty())
    {
      ports_t const& iports = d->input_flow_tag_ports[tag];

      BOOST_FOREACH (port_t const& iport, iports)
      {
        declare_input_port(
          iport,
          new_type,
          info->flags,
          info->description,
          info->frequency);
      }

      ports_t const& oports = d->output_flow_tag_ports[tag];

      BOOST_FOREACH (port_t const& oport, oports)
      {
        declare_output_port(
          oport,
          new_type,
          info->flags,
          info->description,
          info->frequency);
      }

      d->flow_tag_port_types[tag] = new_type;

      return true;
    }
  }

  declare_output_port(
    port,
    new_type,
    info->flags,
    info->description,
    info->frequency);

  return true;
}

config::keys_t
process
::_available_config() const
{
  return config::keys_t();
}

process::conf_info_t
process
::_config_info(config::key_t const& key)
{
  priv::conf_map_t::iterator i = d->config_keys.find(key);

  if (i != d->config_keys.end())
  {
    return i->second;
  }

  throw unknown_configuration_value_exception(d->name, key);
}

void
process
::declare_input_port(port_t const& port, port_info_t const& info)
{
  if (!info)
  {
    throw null_input_port_info_exception(d->name, port);
  }

  port_type_t const& port_type = info->type;

  priv::tag_t const tag = d->port_flow_tag_name(port_type);

  if (!tag.empty())
  {
    d->input_flow_tag_ports[tag].push_back(port);

    d->input_port_tags[port] = tag;

    if (d->flow_tag_port_types[tag])
    {
      port_type_t const& tag_type = *d->flow_tag_port_types[tag];

      declare_input_port(
        port,
        tag_type,
        info->flags,
        info->description,
        info->frequency);

      return;
    }
  }

  port_flags_t const& flags = info->flags;
  port_flags_t::const_iterator i;

  i = flags.find(flag_required);
  bool const required = (i != flags.end());

  i = flags.find(flag_input_static);
  bool const static_ = (i != flags.end());

  i = flags.find(flag_input_nodep);
  bool const no_dep = (i != flags.end());

  if (required && static_)
  {
    /// \todo Throw an exception.
  }

  if (static_)
  {
    declare_configuration_key(
      static_input_prefix + port,
      config::value_t(),
      config::description_t("A default value to use for the \'" + port + "\' port if it is not connected."));

    d->static_inputs.push_back(port);
  }

  if (required && !no_dep)
  {
    d->required_inputs.push_back(port);
  }

  d->input_ports[port] = info;
}

void
process
::declare_input_port(port_t const& port,
                     port_type_t const& type_,
                     port_flags_t const& flags_,
                     port_description_t const& description_,
                     port_frequency_t const& frequency_)
{
  declare_input_port(port, boost::make_shared<port_info>(
    type_,
    flags_,
    description_,
    frequency_));
}

void
process
::declare_output_port(port_t const& port, port_info_t const& info)
{
  if (!info)
  {
    throw null_output_port_info_exception(d->name, port);
  }

  port_type_t const& port_type = info->type;

  priv::tag_t const tag = d->port_flow_tag_name(port_type);

  if (!tag.empty())
  {
    d->output_flow_tag_ports[tag].push_back(port);

    d->output_port_tags[port] = tag;

    if (d->flow_tag_port_types[tag])
    {
      port_type_t const& tag_type = *d->flow_tag_port_types[tag];

      declare_output_port(
        port,
        tag_type,
        info->flags,
        info->description,
        info->frequency);

      return;
    }
  }

  d->output_ports[port] = info;
  d->output_edges[port] = boost::make_shared<priv::output_port_info>();

  port_flags_t const& flags = info->flags;
  port_flags_t::const_iterator const i = flags.find(flag_required);

  if (i != flags.end())
  {
    d->required_outputs.push_back(port);
  }
}

void
process
::declare_output_port(port_t const& port,
                      port_type_t const& type_,
                      port_flags_t const& flags_,
                      port_description_t const& description_,
                      port_frequency_t const& frequency_)
{
  declare_output_port(port, boost::make_shared<port_info>(
    type_,
    flags_,
    description_,
    frequency_));
}

void
process
::set_input_port_frequency(port_t const& port, port_frequency_t const& new_frequency)
{
  if (d->initialized)
  {
    throw set_frequency_on_initialized_process_exception(d->name, port, new_frequency);
  }

  port_info_t const info = input_port_info(port);
  port_frequency_t const& old_frequency = info->frequency;

  if (old_frequency == new_frequency)
  {
    return;
  }

  declare_input_port(
    port,
    info->type,
    info->flags,
    info->description,
    new_frequency);
}

void
process
::set_output_port_frequency(port_t const& port, port_frequency_t const& new_frequency)
{
  if (d->initialized)
  {
    throw set_frequency_on_initialized_process_exception(d->name, port, new_frequency);
  }

  port_info_t const info = output_port_info(port);
  port_frequency_t const& old_frequency = info->frequency;

  if (old_frequency == new_frequency)
  {
    return;
  }

  declare_output_port(
    port,
    info->type,
    info->flags,
    info->description,
    new_frequency);
}

void
process
::remove_input_port(port_t const& port)
{
  // Ensure the port exists.
  priv::port_map_t::const_iterator const p = d->input_ports.find(port);

  if (p == d->input_ports.end())
  {
    throw no_such_port_exception(d->name, port);
  }

  // Remove from known ports.
  d->input_ports.erase(port);

  // Remove all connected edges.
  d->input_edges.erase(port);

  // Remove from bookkeeping structures.
  ports_t::iterator const ri = std::remove(d->required_inputs.begin(), d->required_inputs.end(), port);
  d->required_inputs.erase(ri, d->required_inputs.end());

  priv::port_tag_map_t::const_iterator const t = d->input_port_tags.find(port);

  if (t != d->input_port_tags.end())
  {
    priv::tag_t const& tag = t->second;
    ports_t& ports = d->input_flow_tag_ports[tag];
    ports_t::iterator const i = std::remove(ports.begin(), ports.end(), port);
    ports.erase(i, ports.end());

    if (!ports.size())
    {
      d->check_tag(tag);
    }

    d->input_port_tags.erase(port);
  }
}

void
process
::remove_output_port(port_t const& port)
{
  // Ensure the port exists.
  priv::port_map_t::const_iterator const p = d->output_ports.find(port);

  if (p == d->output_ports.end())
  {
    throw no_such_port_exception(d->name, port);
  }

  // Remove from known ports.
  d->output_ports.erase(port);

  // Remove all connected edges.
  d->output_edges.erase(port);

  // Remove from bookkeeping structures.
  ports_t::iterator const ri = std::remove(d->required_outputs.begin(), d->required_outputs.end(), port);
  d->required_outputs.erase(ri, d->required_outputs.end());

  priv::port_tag_map_t::const_iterator const t = d->output_port_tags.find(port);

  if (t != d->output_port_tags.end())
  {
    priv::tag_t const& tag = t->second;
    ports_t& ports = d->output_flow_tag_ports[tag];
    ports_t::iterator const i = std::remove(ports.begin(), ports.end(), port);
    ports.erase(i, ports.end());

    if (!ports.size())
    {
      d->check_tag(tag);
    }

    d->output_port_tags.erase(port);
  }
}

void
process
::declare_configuration_key(config::key_t const& key, conf_info_t const& info)
{
  if (!info)
  {
    throw null_conf_info_exception(d->name, key);
  }

  d->config_keys[key] = info;
}

void
process
::declare_configuration_key(config::key_t const& key,
                            config::value_t const& def_,
                            config::description_t const& description_)
{
  declare_configuration_key(key, boost::make_shared<conf_info>(
    def_,
    description_));
}

void
process
::mark_process_as_complete()
{
  d->is_complete = true;

  // Indicate to input edges that we are complete.
  BOOST_FOREACH (priv::input_edge_map_t::value_type& port_edge, d->input_edges)
  {
    edge_t& edge = port_edge.second;

    edge->mark_downstream_as_complete();
  }
}

edge_t
process
::input_port_edge(port_t const& port) const
{
  priv::port_map_t::iterator i = d->input_ports.find(port);

  if (i == d->input_ports.end())
  {
    throw no_such_port_exception(d->name, port);
  }

  priv::input_edge_map_t::iterator e = d->input_edges.find(port);

  if (e == d->input_edges.end())
  {
    return edge_t();
  }

  return e->second;
}

edges_t
process
::output_port_edges(port_t const& port) const
{
  priv::port_map_t::iterator i = d->output_ports.find(port);

  if (i == d->output_ports.end())
  {
    throw no_such_port_exception(d->name, port);
  }

  priv::output_edge_map_t::const_iterator const e = d->output_edges.find(port);

  if (e == d->output_edges.end())
  {
    return edges_t();
  }

  priv::output_port_info_t const& info = e->second;
  priv::mutex_t& mut = info->get<0>();

  priv::shared_lock_t const lock(mut);

  (void)lock;

  edges_t const& edges = info->get<1>();

  return edges;
}

edge_datum_t
process
::grab_from_port(port_t const& port) const
{
  priv::port_map_t::iterator i = d->input_ports.find(port);

  if (i == d->input_ports.end())
  {
    throw no_such_port_exception(d->name, port);
  }

  priv::input_edge_map_t::iterator e = d->input_edges.find(port);

  if (e == d->input_edges.end())
  {
    static std::string const reason = "Data was requested from the port";

    throw missing_connection_exception(d->name, port, reason);
  }

  edge_t const& edge = e->second;

  return grab_from_edge(edge);
}

datum_t
process
::grab_datum_from_port(port_t const& port) const
{
  edge_datum_t const edat = grab_from_port(port);

  return edat.datum;
}

void
process
::push_to_port(port_t const& port, edge_datum_t const& dat) const
{
  priv::port_map_t::iterator i = d->output_ports.find(port);

  if (i == d->output_ports.end())
  {
    throw no_such_port_exception(d->name, port);
  }

  priv::output_edge_map_t::const_iterator const e = d->output_edges.find(port);

  if (e != d->output_edges.end())
  {
    priv::output_port_info_t const& info = e->second;
    priv::mutex_t& mut = info->get<0>();

    priv::shared_lock_t const lock(mut);

    (void)lock;

    edges_t const& edges = info->get<1>();

    push_to_edges(edges, dat);
  }
}

void
process
::push_datum_to_port(port_t const& port, datum_t const& dat) const
{
  stamp_t push_stamp;

  {
    priv::output_edge_map_t::iterator const e = d->output_edges.find(port);

    if (e == d->output_edges.end())
    {
      throw no_such_port_exception(d->name, port);
    }

    priv::output_port_info_t& info = e->second;
    priv::mutex_t& mut = info->get<0>();

    priv::upgrade_lock_t lock(mut);

    stamp_t& port_stamp = info->get<2>();

    if (!port_stamp)
    {
      static std::string const reason = "The stamp for an output port was not initialized";

      throw std::runtime_error(reason);
    }

    {
      priv::upgrade_to_unique_lock_t const write_lock(lock);

      (void)write_lock;

      push_stamp = port_stamp;
      port_stamp = stamp::incremented_stamp(port_stamp);
    }
  }

  push_to_port(port, edge_datum_t(dat, push_stamp));
}

config_t
process
::get_config() const
{
  return d->conf;
}

void
process
::ensure_inputs_are_in_sync(bool ensure)
{
  d->input_sync = ensure;
}

void
process
::ensure_inputs_are_valid(bool ensure)
{
  d->input_valid = ensure;
}

process::data_info_t
process
::edge_data_info(edge_data_t const& data)
{
  bool in_sync = true;
  datum::type_t max_type = datum::data;

  edge_datum_t const& fst = data[0];
  stamp_t const& st = fst.stamp;

  BOOST_FOREACH (edge_datum_t const& edat, data)
  {
    datum_t const& dat = edat.datum;
    stamp_t const& st2 = edat.stamp;

    datum::type_t const type = dat->type();

    if (max_type < type)
    {
      max_type = type;
    }

    if (*st != *st2)
    {
      in_sync = false;
    }
  }

  return boost::make_shared<data_info>(in_sync, max_type);
}

void
process
::push_to_edges(edges_t const& edges, edge_datum_t const& dat)
{
  BOOST_FOREACH (edge_t const& edge, edges)
  {
    edge->push_datum(dat);
  }
}

edge_datum_t
process
::grab_from_edge(edge_t const& edge)
{
  return edge->get_datum();
}

edge_datum_t
process
::peek_at_edge(edge_t const& edge)
{
  return edge->peek_datum();
}

config::value_t
process
::config_value_raw(config::key_t const& key) const
{
  priv::conf_map_t::const_iterator const i = d->config_keys.find(key);

  if (i == d->config_keys.end())
  {
    throw unknown_configuration_value_exception(d->name, key);
  }

  if (d->conf->has_value(key))
  {
    return d->conf->get_value<config::value_t>(key);
  }

  conf_info_t const& info = i->second;

  return info->def;
}

bool
process
::is_static_input(port_t const& port) const
{
  ports_t::const_iterator const i = std::find(d->static_inputs.begin(), d->static_inputs.end(), port);

  return (i != d->static_inputs.end());
}

void
process
::set_core_frequency(port_frequency_t const& frequency)
{
  if (!d->initialized)
  {
    static std::string const reason = "A process' frequency was set before it was initialized";

    throw std::runtime_error(reason);
  }

  if (d->core_frequency)
  {
    static std::string const reason = "A process' frequency was set a second time";

    throw std::runtime_error(reason);
  }

  if (frequency.denominator() != 1)
  {
    static std::string const reason = "A process' frequency is not a whole number";

    throw std::runtime_error(reason);
  }

  d->core_frequency = frequency;
}

process::priv
::priv(process* proc, config_t const& c)
  : q(proc)
  , conf(c)
  , configured(false)
  , initialized(false)
  , is_complete(false)
  , input_sync(true)
  , input_valid(true)
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

  q->push_datum_to_port(port_heartbeat, dat);
}

bool
process::priv
::connect_input_port(port_t const& port, edge_t const& edge)
{
  port_map_t::const_iterator const i = input_ports.find(port);

  if (i != input_ports.end())
  {
    if (input_edges[port])
    {
      throw port_reconnect_exception(name, port);
    }

    input_edges[port] = edge;

    return true;
  }

  return false;
}

bool
process::priv
::connect_output_port(port_t const& port, edge_t const& edge)
{
  port_map_t::const_iterator const i = output_ports.find(port);

  if (i != output_ports.end())
  {
    {
      output_port_info_t const& info = output_edges[port];
      priv::mutex_t& mut = info->get<0>();

      priv::unique_lock_t const lock(mut);

      (void)lock;

      edges_t& edges = info->get<1>();

      edges.push_back(edge);
    }

    return true;
  }

  return false;
}

datum_t
process::priv
::check_required_input()
{
  if ((!input_sync && !input_valid) ||
      required_inputs.empty())
  {
    return datum_t();
  }

  edge_data_t first_data;
  edge_data_t data;

  BOOST_FOREACH (port_t const& port, required_inputs)
  {
    input_edge_map_t::const_iterator const i = input_edges.find(port);

    if (i == input_edges.end())
    {
      continue;
    }

    edge_t const& iedge = i->second;

    edge_datum_t const first_edat = peek_at_edge(iedge);
    datum_t const& first_dat = first_edat.datum;
    datum::type_t const first_type = first_dat->type();

    first_data.push_back(first_edat);
    data.push_back(first_edat);

    if ((first_type == datum::flush) ||
        (first_type == datum::complete))
    {
      continue;
    }

    /// \todo Peek N times.
  }

  data_info_t const first_info = edge_data_info(first_data);

  if (input_sync && !first_info->in_sync)
  {
    static datum::error_t const err_string = datum::error_t("Required input edges are not synchronized.");

    return datum::error_datum(err_string);
  }

  // Save the stamp for the inputs.
  if (input_sync)
  {
    stamp_for_inputs = first_data[0].stamp;
  }

  if (!input_valid)
  {
    return datum_t();
  }

  data_info_t const info = edge_data_info(data);

  switch (info->max_status)
  {
    case datum::data:
      break;
    case datum::empty:
      return datum::empty_datum();
    case datum::flush:
      return datum::flush_datum();
    case datum::complete:
      q->mark_process_as_complete();
      return datum::complete_datum();
    case datum::error:
    {
      static datum::error_t const err_string = datum::error_t("Error in a required input edge.");

      return datum::error_datum(err_string);
    }
    case datum::invalid:
    default:
    {
      static datum::error_t const err_string = datum::error_t("Unrecognized datum type in a required input edge.");

      return datum::error_datum(err_string);
    }
  }

  return datum_t();
}

void
process::priv
::grab_from_input_edges()
{
  BOOST_FOREACH (port_map_t::value_type const& iport, input_ports)
  {
    port_t const& port = iport.first;
    port_info_t const& info = iport.second;

    port_frequency_t const& freq = info->frequency;

    if (!freq || (freq.denominator() != 1))
    {
      static std::string const reason = "Cannot automatically pull from "
                                        "an input port with 0 or non-integer "
                                        "frequency";

      throw std::runtime_error(reason);
    }

    frequency_component_t const count = freq.numerator();

    for (frequency_component_t j = 0; j < count; ++j)
    {
      (void)q->grab_datum_from_port(port);
    }
  }
}

void
process::priv
::push_to_output_edges(datum_t const& dat) const
{
  BOOST_FOREACH (port_map_t::value_type const& oport, output_ports)
  {
    port_t const& port = oport.first;
    port_info_t const& info = oport.second;

    port_frequency_t const& freq = info->frequency;

    if (!freq || (freq.denominator() != 1))
    {
      static std::string const reason = "Cannot automatically push to "
                                        "an output port with 0 or non-integer "
                                        "frequency";

      throw std::runtime_error(reason);
    }

    frequency_component_t const count = freq.numerator();

    for (frequency_component_t j = 0; j < count; ++j)
    {
      q->push_datum_to_port(port, dat);
    }
  }
}

bool
process::priv
::required_outputs_done() const
{
  if (required_outputs.empty())
  {
    return false;
  }

  BOOST_FOREACH (port_t const& port, required_outputs)
  {
    output_edge_map_t::const_iterator const i = output_edges.find(port);

    if (i == output_edges.end())
    {
      continue;
    }

    output_port_info_t const& info = i->second;
    priv::mutex_t& mut = info->get<0>();

    priv::shared_lock_t const lock(mut);

    (void)lock;

    edges_t const& edges = info->get<1>();

    BOOST_FOREACH (edge_t const& edge, edges)
    {
      // If any required edge is not complete, then return false.
      if (!edge->is_downstream_complete())
      {
        return false;
      }
    }
  }

  return true;
}

process::priv::tag_t
process::priv
::port_flow_tag_name(port_type_t const& port_type) const
{
  if (boost::starts_with(port_type, type_flow_dependent))
  {
    return port_type.substr(type_flow_dependent.size());
  }

  return tag_t();
}

void
process::priv
::check_tag(tag_t const& tag)
{
  ports_t const iports = input_flow_tag_ports[tag];
  ports_t const oports = output_flow_tag_ports[tag];

  if (!iports.size() && !oports.size())
  {
    flow_tag_port_types.erase(tag);
  }
}

}
