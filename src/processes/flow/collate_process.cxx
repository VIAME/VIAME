/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "collate_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>

#include <map>
#include <string>

/**
 * \file collate_process.cxx
 *
 * \brief Implementation of the collate process.
 */

namespace vistk
{

class collate_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t tag_t;

    struct tag_info
    {
      ports_t ports;
      ports_t::const_iterator cur_port;
    };
    typedef std::map<tag_t, tag_info> tag_data_t;

    tag_data_t tag_data;

    tag_t tag_for_coll_port(port_t const& port) const;

    static port_t const res_sep;
    static port_t const port_res_prefix;
    static port_t const port_color_prefix;
    static port_t const port_coll_prefix;
};

process::port_t const collate_process::priv::res_sep = port_t("/");
process::port_t const collate_process::priv::port_res_prefix = port_t("res") + res_sep;
process::port_t const collate_process::priv::port_color_prefix = port_t("color") + res_sep;
process::port_t const collate_process::priv::port_coll_prefix = port_t("coll") + res_sep;

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
 *   \termdef{The type of the port. This must be one of \type{res},
 *   \type{color}, or \type{coll}.}
 * \term{\portvar{tag}}
 *   \termdef{The name of the stream the port is associated with.}
 * \term{\portvar{group}}
 *   \termdef{Only required for \type{coll}-type ports. Data from the same
 *   \portvar{tag} stream from its \type{res} port is collected in sorted order
 *   over all of the \type{coll} ports.}
 * </dl>
 *
 * The available port types are:
 *
 * <dl>
 * \term{\type{color}}
 *   \termdef{This is the trigger port for the associated tagged stream. When
 *   this port for the given tag is connected to, the \type{res} and \type{coll}
 *   ports for the tag will not cause errors. The stamp received on the port
 *   represents will be applied to the data leaving the \type{res} port for the
 *   tag.}
 * \term{\type{res}}
 *   \termdef{This port for the given tag is where the data for a stream leaves
 *   the process. The stamp from the \type{color} port for the \portvar{tag} is
 *   applied.}
 * \term{\type{coll}}
 *   \termdef{These ports for a given \portvar{tag} receive data from a set of
 *   sources, likely made by the \ref distribute_process. Data is collected in
 *   sorted ordef of the \type{group} name, combined with the next value from
 *   the \type{color} port and sent out the \type{res} port for the
 *   \portvar{tag}.}
 * </dl>
 */

collate_process
::collate_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  // This process manages its own colors and inputs.
  ensure_inputs_are_same_color(false);
  ensure_inputs_are_valid(false);
}

collate_process
::~collate_process()
{
}

void
collate_process
::_init()
{
  BOOST_FOREACH (priv::tag_data_t::value_type& tag_data, d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    priv::tag_info& info = tag_data.second;
    ports_t const& ports = info.ports;

    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to collate "
                                 "to for the \"" + tag + "\" result data";

      throw invalid_configuration_exception(name(), reason);
    }

    frequency_component_t const ratio = ports.size();
    port_frequency_t const freq = port_frequency_t(1, ratio);

    BOOST_FOREACH (port_t const& port, ports)
    {
      set_input_port_frequency(port, freq);
    }

    info.cur_port = ports.begin();
  }

  process::_init();
}

void
collate_process
::_reset()
{
  BOOST_FOREACH (priv::tag_data_t::value_type const& tag_data, d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const output_port = priv::port_res_prefix + tag;
    port_t const color_port = priv::port_color_prefix + tag;
    priv::tag_info const& info = tag_data.second;
    ports_t const& ports = info.ports;

    BOOST_FOREACH (port_t const& port, ports)
    {
      remove_input_port(port);
    }

    remove_input_port(color_port);
    remove_output_port(output_port);
  }

  d->tag_data.clear();

  process::_reset();
}

void
collate_process
::_step()
{
  ports_t complete_ports;

  BOOST_FOREACH (priv::tag_data_t::value_type& tag_data, d->tag_data)
  {
    priv::tag_t const& tag = tag_data.first;
    port_t const output_port = priv::port_res_prefix + tag;
    port_t const color_port = priv::port_color_prefix + tag;
    priv::tag_info& info = tag_data.second;

    edge_datum_t const coll_dat = grab_from_port(*info.cur_port);
    stamp_t const coll_stamp = coll_dat.get<1>();

    ++info.cur_port;

    if (info.cur_port == info.ports.end())
    {
      info.cur_port = info.ports.begin();
    }

    edge_datum_t const color_dat = grab_from_port(color_port);
    stamp_t const color_stamp = color_dat.get<1>();

    edge_data_t data;

    data.push_back(coll_dat);
    data.push_back(color_dat);

    if (edge_data_info(data)->max_status == datum::complete)
    {
      push_to_port(output_port, edge_datum_t(datum::complete_datum(), color_stamp));

      complete_ports.push_back(tag);

      continue;
    }

    edge_datum_t res_dat = coll_dat;
    boost::get<1>(res_dat) = stamp::recolored_stamp(coll_stamp, color_stamp);

    push_to_port(output_port, res_dat);
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

process::constraints_t
collate_process
::_constraints() const
{
  constraints_t consts = process::_constraints();

  consts.insert(constraint_unsync_input);

  return consts;
}

process::port_info_t
collate_process
::_input_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_color_prefix))
  {
    priv::tag_t const tag = port.substr(priv::port_color_prefix.size());

    priv::tag_data_t::const_iterator const i = d->tag_data.find(tag);

    if (i == d->tag_data.end())
    {
      priv::tag_info info;

      d->tag_data[tag] = info;

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(
        port,
        type_none,
        required,
        port_description_t("The original color for the result " + tag + "."));
      declare_output_port(
        priv::port_res_prefix + tag,
        type_flow_dependent + tag,
        required,
        port_description_t("The output port for " + tag + "."));
    }
  }

  priv::tag_t const tag = d->tag_for_coll_port(port);

  if (!tag.empty())
  {
    priv::tag_info& info = d->tag_data[tag];

    info.ports.push_back(port);

    port_flags_t required;

    required.insert(flag_required);

    declare_input_port(
      port,
      type_flow_dependent + tag,
      required,
      port_description_t("An input for the " + tag + " data."));
  }

  return process::_input_port_info(port);
}

collate_process::priv
::priv()
{
}

collate_process::priv
::~priv()
{
}

collate_process::priv::tag_t
collate_process::priv
::tag_for_coll_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_coll_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_coll_prefix.size());

    BOOST_FOREACH (priv::tag_data_t::value_type const& data, tag_data)
    {
      tag_t const& tag = data.first;
      port_t const tag_prefix = tag + priv::res_sep;

      if (boost::starts_with(no_prefix, tag_prefix))
      {
        return tag;
      }
    }
  }

  return tag_t();
}

}
