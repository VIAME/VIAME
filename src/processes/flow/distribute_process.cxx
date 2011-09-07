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

    typedef std::map<port_t, stamp_t> dist_ports_t;
    struct dist_info
    {
      stamp_t stamp_color;
      dist_ports_t dist_ports;
      dist_ports_t::const_iterator cur_port;
    };
    typedef std::map<port_t, dist_info> dist_data_t;

    dist_data_t dist_data;

    port_t src_for_dist_port(port_t const& port) const;

    static port_t const src_sep;
    static port_t const port_src_prefix;
    static port_t const port_color_prefix;
    static port_t const port_dist_prefix;
};

process::port_t const distribute_process::priv::src_sep = port_t("_");
process::port_t const distribute_process::priv::port_src_prefix = port_t("src") + src_sep;
process::port_t const distribute_process::priv::port_color_prefix = port_t("color") + src_sep;
process::port_t const distribute_process::priv::port_dist_prefix = port_t("dist") + src_sep;

distribute_process
::distribute_process(config_t const& config)
  : process(config)
{
  d.reset(new priv);
}

distribute_process
::~distribute_process()
{
}

void
distribute_process
::_init()
{
  BOOST_FOREACH (priv::dist_data_t::value_type& dist_data, d->dist_data)
  {
    priv::dist_info& info = dist_data.second;
    priv::dist_ports_t const& ports = info.dist_ports;

    if (ports.size() < 2)
    {
      std::string const reason = "There must be at least two ports to distribute "
                                 "to for the " + dist_data.first + " source data";

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

  BOOST_FOREACH (priv::dist_data_t::value_type& dist_data, d->dist_data)
  {
    port_t const input_port = priv::port_src_prefix + dist_data.first;
    port_t const color_port = priv::port_color_prefix + dist_data.first;
    priv::dist_info& info = dist_data.second;

    edge_datum_t const src_dat = grab_from_port(input_port);
    stamp_t const src_stamp = src_dat.get<1>();

    if (src_dat.get<0>()->type() == datum::complete)
    {
      push_to_port(color_port, edge_datum_t(datum::complete_datum(), src_stamp));

      BOOST_FOREACH (priv::dist_ports_t::value_type const& port, info.dist_ports)
      {
        stamp_t const dist_stamp = stamp::recolored_stamp(src_stamp, port.second);

        push_to_port(port.first, edge_datum_t(datum::complete_datum(), dist_stamp));
      }

      complete_ports.push_back(dist_data.first);

      continue;
    }

    push_to_port(color_port, edge_datum_t(datum::empty_datum(), src_dat.get<1>()));

    edge_datum_t dist_dat = src_dat;
    boost::get<1>(dist_dat) = stamp::recolored_stamp(src_stamp, info.cur_port->second);

    push_to_port(info.cur_port->first, dist_dat);

    ++info.cur_port;

    if (info.cur_port == info.dist_ports.end())
    {
      info.cur_port = info.dist_ports.begin();
    }
  }

  BOOST_FOREACH (port_t const& port, complete_ports)
  {
    d->dist_data.erase(port);
  }

  if (d->dist_data.empty())
  {
    mark_as_complete();
  }
}

process::port_info_t
distribute_process
::_output_port_info(port_t const& port)
{
  if (boost::starts_with(port, priv::port_color_prefix))
  {
    port_t const src_name = port.substr(priv::port_color_prefix.size());

    priv::dist_data_t::const_iterator const i = d->dist_data.find(src_name);

    if (i == d->dist_data.end())
    {
      priv::dist_info info;

      info.stamp_color = stamp::new_stamp();

      d->dist_data[src_name] = info;

      port_flags_t required;

      required.insert(flag_required);

      declare_input_port(priv::port_src_prefix + src_name, boost::make_shared<port_info>(
        type_any,
        required,
        port_description_t("The input port for " + src_name + ".")));
      declare_output_port(port, boost::make_shared<port_info>(
        type_none,
        required,
        port_description_t("The original color for the input " + src_name + ".")));
    }
  }

  port_t const src_for_dist = d->src_for_dist_port(port);

  if (!src_for_dist.empty())
  {
    d->dist_data[src_for_dist].dist_ports[port] = stamp::new_stamp();

    port_flags_t required;

    required.insert(flag_required);

    declare_output_port(port, boost::make_shared<port_info>(
      type_any,
      required,
      port_description_t("An output for the " + src_for_dist + " data.")));
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
::src_for_dist_port(port_t const& port) const
{
  if (boost::starts_with(port, priv::port_dist_prefix))
  {
    port_t const no_prefix = port.substr(priv::port_dist_prefix.size());

    BOOST_FOREACH (priv::dist_data_t::value_type const& data, dist_data)
    {
      port_t const src_prefix = data.first + priv::src_sep;

      if (boost::starts_with(no_prefix, src_prefix))
      {
        return data.first;
      }
    }
  }

  return port_t();
}

}
