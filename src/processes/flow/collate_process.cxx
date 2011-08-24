/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "collate_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/foreach.hpp>

#include <map>

namespace vistk
{

class collate_process::priv
{
  public:
    priv();
    ~priv();

    struct coll_info
    {
      ports_t coll_ports;
      ports_t::const_iterator cur_port;
    };
    typedef std::map<port_t, coll_info> coll_data_t;

    coll_data_t coll_data;

    port_t res_for_coll_port(port_t const& port) const;

    static port_t const res_sep;
    static port_t const port_res_prefix;
    static port_t const port_color_prefix;
    static port_t const port_coll_prefix;
};

process::port_t const collate_process::priv::res_sep = port_t("_");
process::port_t const collate_process::priv::port_res_prefix = port_t("res") + res_sep;
process::port_t const collate_process::priv::port_color_prefix = port_t("color") + res_sep;
process::port_t const collate_process::priv::port_coll_prefix = port_t("coll") + res_sep;

collate_process
::collate_process(config_t const& config)
  : process(config)
{
  d = boost::shared_ptr<priv>(new priv);
}

collate_process
::~collate_process()
{
}

void
collate_process
::_init()
{
  BOOST_FOREACH (priv::coll_data_t::value_type& coll_data, d->coll_data)
  {
    priv::coll_info& info = coll_data.second;
    ports_t const& ports = info.coll_ports;

    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to collate "
                                 "to for the " + coll_data.first + " result data";

      throw invalid_configuration_exception(name(), reason);
    }

    info.cur_port = ports.begin();
  }
}

void
collate_process
::_step()
{
  ports_t complete_ports;

  BOOST_FOREACH (priv::coll_data_t::value_type& coll_data, d->coll_data)
  {
    port_t const output_port = priv::port_res_prefix + coll_data.first;
    port_t const color_port = priv::port_color_prefix + coll_data.first;
    priv::coll_info& info = coll_data.second;

    edge_datum_t const coll_dat = grab_from_port(*info.cur_port);
    stamp_t const coll_stamp = coll_dat.get<1>();

    ++info.cur_port;

    if (info.cur_port == info.coll_ports.end())
    {
      info.cur_port = info.coll_ports.begin();
    }

    edge_datum_t const color_dat = grab_from_port(color_port);
    stamp_t const color_stamp = color_dat.get<1>();

    edge_data_t data;

    data.push_back(coll_dat);
    data.push_back(color_dat);

    if (edge_data_info(data)->max_status == datum::DATUM_COMPLETE)
    {
      push_to_port(output_port, edge_datum_t(datum::complete_datum(), color_stamp));

      complete_ports.push_back(coll_data.first);

      continue;
    }

    edge_datum_t res_dat = coll_dat;
    boost::get<1>(res_dat) = stamp::recolored_stamp(coll_stamp, color_stamp);

    push_to_port(output_port, res_dat);
  }

  BOOST_FOREACH (port_t const& port, complete_ports)
  {
    d->coll_data.erase(port);
  }

  if (d->coll_data.empty())
  {
    mark_as_complete();
  }
}

void
collate_process
::_connect_input_port(port_t const& port, edge_ref_t edge)
{
  if (boost::starts_with(port, priv::port_color_prefix))
  {
    port_t const res_name = port.substr(priv::port_color_prefix.size());

    priv::coll_data_t::const_iterator const i = d->coll_data.find(res_name);

    if (i != d->coll_data.end())
    {
      throw port_reconnect_exception(name(), port);
    }

    d->coll_data[res_name] = priv::coll_info();

    port_flags_t required;

    required.insert(flag_required);

    declare_input_port(port, port_info_t(new port_info(
      type_none,
      required,
      port_description_t("The original color for the result " + res_name + "."))));
    declare_output_port(priv::port_res_prefix + res_name, port_info_t(new port_info(
      type_any,
      required,
      port_description_t("The output port for " + res_name + "."))));
  }

  port_t const res_for_dist = d->res_for_coll_port(port);

  if (!res_for_dist.empty())
  {
    d->coll_data[res_for_dist].coll_ports.push_back(port);

    port_flags_t required;

    required.insert(flag_required);

    declare_input_port(port, port_info_t(new port_info(
      type_any,
      required,
      port_description_t("An input for the " + res_for_dist + " data."))));
  }

  process::_connect_input_port(port, edge);
}

collate_process::priv
::priv()
{
}

collate_process::priv
::~priv()
{
}

process::port_t
collate_process::priv
::res_for_coll_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_coll_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_coll_prefix.size());

    BOOST_FOREACH (priv::coll_data_t::value_type const& data, coll_data)
    {
      port_t const res_prefix = data.first + priv::res_sep;

      if (boost::starts_with(no_prefix, res_prefix))
      {
        return data.first;
      }
    }
  }

  return port_t();
}

}
