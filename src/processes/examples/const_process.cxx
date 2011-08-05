/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "const_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>

namespace vistk
{

class const_process::priv
{
  public:
    priv();
    ~priv();

    edges_t output_edges;

    port_info_t output_port_info;

    static port_t const OUTPUT_PORT_NAME;
};

process::port_t const const_process::priv::OUTPUT_PORT_NAME = process::port_t("const");

const_process
::const_process(config_t const& config)
  : process(config)
{
  d = boost::shared_ptr<priv>(new priv);
}

const_process
::~const_process()
{
}

void
const_process
::_step()
{
  edge_datum_t const edat = edge_datum_t(datum::empty_datum(), heartbeat_stamp());

  push_to_edges(d->output_edges, edat);

  process::_step();
}

void
const_process
::_connect_output_port(port_t const& port, edge_t edge)
{
  if (port == priv::OUTPUT_PORT_NAME)
  {
    d->output_edges.push_back(edge);

    return;
  }

  process::_connect_output_port(port, edge);
}

process::port_info_t
const_process
::_output_port_info(port_t const& port) const
{
  if (port == priv::OUTPUT_PORT_NAME)
  {
    return d->output_port_info;
  }

  return process::_output_port_info(port);
}

process::ports_t
const_process
::_output_ports() const
{
  ports_t ports;

  ports.push_back(priv::OUTPUT_PORT_NAME);

  return ports;
}

const_process::priv
::priv()
{
  port_flags_t const_required;

  const_required.insert(flag_required);
  const_required.insert(flag_output_const);

  output_port_info = port_info_t(new port_info(
    type_none,
    const_required,
    port_description_t("The port with the const flag set.")));
}

const_process::priv
::~priv()
{
}

}
