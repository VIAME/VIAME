/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "distribute_process.h"

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/process_exception.h>
#include <sprokit/pipeline/stamp.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <string>

/**
 * \file distribute_process.cxx
 *
 * \brief Implementation of the distribute process.
 */

namespace sprokit
{

class distribute_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t group_t;
    typedef port_t tag_t;

    class tag_info
    {
      public:
        tag_info();
        ~tag_info();

        ports_t ports;
        ports_t::const_iterator cur_port;
    };
    typedef std::map<tag_t, tag_info> tag_data_t;

    tag_data_t tag_data;

    // Port name break down.
    tag_t tag_for_dist_port(port_t const& port) const;
    group_t group_for_dist_port(port_t const& port) const;

    static port_t const src_sep;
    static port_t const port_src_prefix;
    static port_t const port_status_prefix;
    static port_t const port_dist_prefix;
};

process::port_t const distribute_process::priv::src_sep = port_t("/");
process::port_t const distribute_process::priv::port_src_prefix = port_t("src") + src_sep;
process::port_t const distribute_process::priv::port_status_prefix = port_t("status") + src_sep;
process::port_t const distribute_process::priv::port_dist_prefix = port_t("dist") + src_sep;

/**
 * \internal
 *
 * Ports on the \ref distribute_process are broken down as follows:
 *
 *   \portvar{type}/\portvar{tag}[/\portvar{group}]
 *
 * The port name is broken down as follows:
 *
 * <dl>
 * \term{\portvar{type}}
 *   \termdef{The type of the port. This must be one of \type{src},
 *   \type{status}, or \type{dist}.}
 * \term{\portvar{tag}}
 *   \termdef{The name of the stream the port is associated with.}
 * \term{\portvar{group}}
 *   \termdef{Only required for \type{dist}-type ports. Data from the same
 *   \portvar{tag} stream from its \type{src} port is distributed in sorted
 *   order over all of the \type{dist} ports.}
 * </dl>
 *
 * The available port types are:
 *
 * <dl>
 * \term{\type{status}}
 *   \termdef{This is the trigger port for the associated tagged stream. When
 *   this port for the given tag is connected to, the \type{src} and \type{dist}
 *   ports for the tag will not cause errors. The stamp on the port represents
 *   the status of the original stream coming in the \type{src} port for the
 *   tag.}
 * \term{\type{src}}
 *   \termdef{This port for the given tag is where the original stream comes
 *   into the process.}
 * \term{\type{dist}}
 *   \termdef{These ports for a given \portvar{tag} receive a subset of the data
 *   from the corresponding \type{src} port. They are used in sorted order of
 *   the \type{group} name.}
 * </dl>
 */

distribute_process
::distribute_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  // This process manages its own inputs.
  set_data_checking_level(check_none);
}

distribute_process
::~distribute_process()
{
}

void
distribute_process
::_init()
{
  BOOST_FOREACH (priv::tag_data_t::value_type& tag_data, d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    priv::tag_info& info = tag_data.second;
    ports_t const& ports = info.ports;

    // Ensure that the extra process is actually doing work.
    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to distribute "
                                 "to for the \"" + tag + "\" source data";

      throw invalid_configuration_exception(name(), reason);
    }

    frequency_component_t const ratio = ports.size();
    port_frequency_t const freq = port_frequency_t(1, ratio);

    BOOST_FOREACH (port_t const& port, ports)
    {
      set_output_port_frequency(port, freq);
    }

    info.cur_port = ports.begin();
  }

  process::_init();
}

void
distribute_process
::_reset()
{
  BOOST_FOREACH (priv::tag_data_t::value_type const& tag_data, d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const input_port = priv::port_src_prefix + tag;
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info const& info = tag_data.second;
    ports_t const& ports = info.ports;

    BOOST_FOREACH (port_t const& port, ports)
    {
      remove_output_port(port);
    }

    remove_output_port(status_port);
    remove_input_port(input_port);
  }

  d->tag_data.clear();

  process::_reset();
}

void
distribute_process
::_step()
{
  ports_t complete_ports;

  BOOST_FOREACH (priv::tag_data_t::value_type& tag_data, d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const input_port = priv::port_src_prefix + tag;
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info& info = tag_data.second;

    edge_datum_t const src_edat = grab_from_port(input_port);
    datum_t const& src_dat = src_edat.datum;
    stamp_t const& src_stamp = src_edat.stamp;

    datum::type_t const src_type = src_dat->type();

    bool const is_complete = (src_type == datum::complete);

    if (is_complete || (src_type == datum::flush))
    {
      push_to_port(status_port, src_edat);

      BOOST_FOREACH (port_t const& port, info.ports)
      {
        push_to_port(port, src_edat);
      }

      if (is_complete)
      {
        complete_ports.push_back(tag);

        continue;
      }
    }
    else
    {
      edge_datum_t const status_edat = edge_datum_t(datum::empty_datum(), src_stamp);

      push_to_port(status_port, status_edat);
      push_to_port(*info.cur_port, src_edat);
    }

    ++info.cur_port;

    if (info.cur_port == info.ports.end())
    {
      info.cur_port = info.ports.begin();
    }
  }

  BOOST_FOREACH (port_t const& port, complete_ports)
  {
    d->tag_data.erase(port);
  }

  if (d->tag_data.empty())
  {
    mark_process_as_complete();
  }

  process::_step();
}

process::properties_t
distribute_process
::_properties() const
{
  properties_t consts = process::_properties();

  consts.insert(property_unsync_output);

  return consts;
}

process::port_info_t
distribute_process
::_output_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_status_prefix))
  {
    priv::tag_t const tag = port.substr(priv::port_status_prefix.size());

    if (!d->tag_data.count(tag))
    {
      priv::tag_info info;

      d->tag_data[tag] = info;

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(
        priv::port_src_prefix + tag,
        type_flow_dependent + tag,
        required,
        port_description_t("The input port for " + tag + "."));
      declare_output_port(
        port,
        type_none,
        required,
        port_description_t("The status for the input " + tag + "."));
    }
  }

  priv::tag_t const tag = d->tag_for_dist_port(port);

  if (!tag.empty())
  {
    port_t const group = d->group_for_dist_port(port);
    priv::tag_info& info = d->tag_data[tag];

    info.ports.push_back(port);

    port_flags_t required;

    required.insert(flag_required);

    declare_output_port(
      port,
      type_flow_dependent + tag,
      required,
      port_description_t("An output for the " + tag + " data."));
  }

  return process::_output_port_info(port);
}

distribute_process::priv
::priv()
  : tag_data()
{
}

distribute_process::priv
::~priv()
{
}

distribute_process::priv::tag_t
distribute_process::priv
::tag_for_dist_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_dist_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_dist_prefix.size());

    BOOST_FOREACH (priv::tag_data_t::value_type const& data, tag_data)
    {
      tag_t const& tag = data.first;
      port_t const tag_prefix = tag + priv::src_sep;

      if (boost::starts_with(no_prefix, tag_prefix))
      {
        return tag;
      }
    }
  }

  return tag_t();
}

distribute_process::priv::group_t
distribute_process::priv
::group_for_dist_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_dist_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_dist_prefix.size());

    BOOST_FOREACH (priv::tag_data_t::value_type const& data, tag_data)
    {
      tag_t const& tag = data.first;
      port_t const tag_prefix = tag + priv::src_sep;

      if (boost::starts_with(no_prefix, tag_prefix))
      {
        group_t const group = no_prefix.substr(tag_prefix.size());

        return group;
      }
    }
  }

  return group_t();
}

distribute_process::priv::tag_info
::tag_info()
  : ports()
  , cur_port()
{
}

distribute_process::priv::tag_info
::~tag_info()
{
}

}
