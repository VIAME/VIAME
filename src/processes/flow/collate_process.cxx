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

    tag_t tag_for_coll_port(port_t const& port) const;

    static port_t const res_sep;
    static port_t const port_res_prefix;
    static port_t const port_status_prefix;
    static port_t const port_coll_prefix;
};

process::port_t const collate_process::priv::res_sep = port_t("/");
process::port_t const collate_process::priv::port_res_prefix = port_t("res") + res_sep;
process::port_t const collate_process::priv::port_status_prefix = port_t("status") + res_sep;
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
 *   \type{status}, or \type{coll}.}
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
 * \term{\type{status}}
 *   \termdef{This is the trigger port for the associated tagged stream. When
 *   this port for the given tag is connected to, the \type{res} and \type{coll}
 *   ports for the tag will not cause errors.}
 * \term{\type{res}}
 *   \termdef{This port for the given tag is where the data for a stream leaves
 *   the process. The stamp from the \type{status} port for the \portvar{tag} is
 *   applied.}
 * \term{\type{coll}}
 *   \termdef{These ports for a given \portvar{tag} receive data from a set of
 *   sources, likely made by the \ref distribute_process. Data is collected in
 *   sorted ordef of the \type{group} name and sent out the \type{res} port for
 *   the \portvar{tag}.}
 * </dl>
 */

collate_process
::collate_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  // This process manages its own inputs.
  set_data_checking_level(check_none);
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
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info const& info = tag_data.second;
    ports_t const& ports = info.ports;

    BOOST_FOREACH (port_t const& port, ports)
    {
      remove_input_port(port);
    }

    remove_input_port(status_port);
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
    port_t const status_port = priv::port_status_prefix + tag;
    priv::tag_info& info = tag_data.second;

    edge_datum_t const status_edat = grab_from_port(status_port);
    datum_t const& status_dat = status_edat.datum;

    datum::type_t const status_type = status_dat->type();

    bool const is_complete = (status_type == datum::complete);

    if (is_complete || (status_type == datum::flush))
    {
      push_to_port(output_port, status_edat);

      BOOST_FOREACH (port_t const& port, info.ports)
      {
        (void)grab_from_port(port);
      }

      if (is_complete)
      {
        complete_ports.push_back(tag);

        continue;
      }
    }
    else
    {
      edge_datum_t const coll_dat = grab_from_port(*info.cur_port);

      push_to_port(output_port, coll_dat);
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
collate_process
::_properties() const
{
  properties_t consts = process::_properties();

  consts.insert(property_unsync_input);

  return consts;
}

process::port_info_t
collate_process
::_input_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_status_prefix))
  {
    priv::tag_t const tag = port.substr(priv::port_status_prefix.size());

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
        port_description_t("The original status for the result " + tag + "."));
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
  : tag_data()
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

collate_process::priv::tag_info
::tag_info()
  : ports()
  , cur_port()
{
}

collate_process::priv::tag_info
::~tag_info()
{
}

}
