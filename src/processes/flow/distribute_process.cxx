/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "distribute_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>
#include <boost/make_shared.hpp>

#include <map>
#include <string>

/**
 * \file distribute_process.cxx
 *
 * \brief Implementation of the distribute process.
 */

namespace vistk
{

class distribute_process::priv
{
  public:
    priv();
    ~priv();

    typedef std::map<port_t, stamp_t> group_colors_t;
    typedef std::map<port_t, stamp_t> tag_ports_t;
    struct tag_info
    {
      tag_ports_t ports;
      tag_ports_t::const_iterator cur_port;
    };
    typedef std::map<port_t, tag_info> tag_data_t;

    tag_data_t tag_data;
    group_colors_t group_colors;

    // Port name break down.
    port_t tag_for_dist_port(port_t const& port) const;
    port_t group_for_dist_port(port_t const& port) const;

    stamp_t color_for_group(port_t const& group);

    static port_t const src_sep;
    static port_t const port_src_prefix;
    static port_t const port_color_prefix;
    static port_t const port_dist_prefix;
};

process::port_t const distribute_process::priv::src_sep = port_t("/");
process::port_t const distribute_process::priv::port_src_prefix = port_t("src") + src_sep;
process::port_t const distribute_process::priv::port_color_prefix = port_t("color") + src_sep;
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
 *   \type{color}, or \type{dist}.}
 * \term{\portvar{tag}}
 *   \termdef{The name of the stream the port is associated with.}
 * \term{\portvar{group}}
 *   \termdef{Only required for \type{dist}-type ports. Data from the same
 *   \portvar{tag} stream from its \type{src} port is distributed in sorted
 *   order over all of the \type{dist} ports. Ports which share the same
 *   \portvar{group} name will share a common stamp stream. Each \portvar{group}
 *   will receive different colorings on the stamps to avoid mixing data.}
 * </dl>
 *
 * The available port types are:
 *
 * <dl>
 * \term{\type{color}}
 *   \termdef{This is the trigger port for the associated tagged stream. When
 *   this port for the given tag is connected to, the \type{src} and \type{dist}
 *   ports for the tag will not cause errors. The stamp on the port represents
 *   the original stamps coming in the \type{src} port for the tag.}
 * \term{\type{src}}
 *   \termdef{This port for the given tag is where the original stream comes
 *   into the process. The stamp is stripped off and sent down the corresponding
 *   \type{color} port while the data is distributed among the \type{dist}
 *   ports in distinct colored streams.}
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
  // This process manages its own colors and inputs.
  ensure_inputs_are_same_color(false);
  ensure_inputs_are_valid(false);
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
    port_t const& tag = tag_data.first;
    priv::tag_info& info = tag_data.second;
    priv::tag_ports_t const& ports = info.ports;

    // Ensure that the extra process is actually doing work.
    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to distribute "
                                 "to for the \"" + tag + "\" source data";

      throw invalid_configuration_exception(name(), reason);
    }

    info.cur_port = ports.begin();
  }
}

void
distribute_process
::_step()
{
  ports_t complete_ports;

  BOOST_FOREACH (priv::tag_data_t::value_type& tag_data, d->tag_data)
  {
    port_t const& tag = tag_data.first;
    port_t const input_port = priv::port_src_prefix + tag;
    port_t const color_port = priv::port_color_prefix + tag;
    priv::tag_info& info = tag_data.second;

    edge_datum_t const src_dat = grab_from_port(input_port);
    stamp_t const src_stamp = src_dat.get<1>();

    if (src_dat.get<0>()->type() == datum::complete)
    {
      push_to_port(color_port, edge_datum_t(datum::complete_datum(), src_stamp));

      BOOST_FOREACH (priv::tag_ports_t::value_type const& tag_port, info.ports)
      {
        port_t const& dist_port = tag_port.first;
        stamp_t const& group_stamp = tag_port.second;
        stamp_t const dist_stamp = stamp::recolored_stamp(src_stamp, group_stamp);

        push_to_port(dist_port, edge_datum_t(datum::complete_datum(), dist_stamp));
      }

      complete_ports.push_back(tag);

      continue;
    }

    push_to_port(color_port, edge_datum_t(datum::empty_datum(), src_dat.get<1>()));

    edge_datum_t dist_dat = src_dat;
    boost::get<1>(dist_dat) = stamp::recolored_stamp(src_stamp, info.cur_port->second);

    push_to_port(info.cur_port->first, dist_dat);

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
}

process::constraints_t
distribute_process
::_constraints() const
{
  constraints_t consts = process::_constraints();

  consts.insert(constraint_unsync_output);

  return consts;
}

process::port_info_t
distribute_process
::_output_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_color_prefix))
  {
    port_t const tag = port.substr(priv::port_color_prefix.size());

    priv::tag_data_t::const_iterator const i = d->tag_data.find(tag);

    if (i == d->tag_data.end())
    {
      priv::tag_info info;

      d->tag_data[tag] = info;

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(priv::port_src_prefix + tag, boost::make_shared<port_info>(
        type_any,
        required,
        port_description_t("The input port for " + tag + ".")));
      declare_output_port(port, boost::make_shared<port_info>(
        type_none,
        required,
        port_description_t("The original color for the input " + tag + ".")));
    }
  }

  port_t const tag = d->tag_for_dist_port(port);

  if (!tag.empty())
  {
    port_t const group = d->group_for_dist_port(port);
    priv::tag_info& info = d->tag_data[tag];

    info.ports[port] = d->color_for_group(group);

    port_flags_t required;

    required.insert(flag_required);

    declare_output_port(port, boost::make_shared<port_info>(
      type_any,
      required,
      port_description_t("An output for the " + tag + " data.")));
  }

  return process::_output_port_info(port);
}

distribute_process::priv
::priv()
{
}

distribute_process::priv
::~priv()
{
}

process::port_t
distribute_process::priv
::tag_for_dist_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_dist_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_dist_prefix.size());

    BOOST_FOREACH (priv::tag_data_t::value_type const& data, tag_data)
    {
      port_t const& tag = data.first;
      port_t const tag_prefix = tag + priv::src_sep;

      if (boost::starts_with(no_prefix, tag_prefix))
      {
        return tag;
      }
    }
  }

  return port_t();
}

process::port_t
distribute_process::priv
::group_for_dist_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_dist_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_dist_prefix.size());

    BOOST_FOREACH (priv::tag_data_t::value_type const& data, tag_data)
    {
      port_t const& tag = data.first;
      port_t const tag_prefix = tag + priv::src_sep;

      if (boost::starts_with(no_prefix, tag_prefix))
      {
        port_t const group = no_prefix.substr(tag_prefix.size());

        return group;
      }
    }
  }

  return port_t();
}

stamp_t
distribute_process::priv
::color_for_group(port_t const& group)
{
  group_colors_t::const_iterator const i = group_colors.find(group);

  if (i == group_colors.end())
  {
    group_colors[group] = stamp::new_stamp();
  }

  return group_colors[group];
}

}
