/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "mutate_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

namespace vistk
{

class mutate_process::priv
{
  public:
    priv();
    ~priv();

    edge_ref_t input_edge;

    port_info_t input_port_info;

    static port_t const INPUT_PORT_NAME;
};

process::port_t const mutate_process::priv::INPUT_PORT_NAME = process::port_t("mutate");

mutate_process
::mutate_process(config_t const& config)
  : process(config)
{
  d = boost::shared_ptr<priv>(new priv);
}

mutate_process
::~mutate_process()
{
}

void
mutate_process
::_step()
{
  edge_datum_t const input_dat = grab_from_edge_ref(d->input_edge);

  switch (input_dat.get<0>()->type())
  {
    case datum::DATUM_DATA:
      break;
    case datum::DATUM_EMPTY:
      break;
    case datum::DATUM_COMPLETE:
      mark_as_complete();
      break;
    case datum::DATUM_ERROR:
      break;
    case datum::DATUM_INVALID:
    default:
      break;
  }

  process::_step();
}

void
mutate_process
::_connect_input_port(port_t const& port, edge_ref_t edge)
{
  if (port == priv::INPUT_PORT_NAME)
  {
    if (d->input_edge.use_count())
    {
      throw port_reconnect_exception(name(), port);
    }

    d->input_edge = edge;

    return;
  }

  process::_connect_input_port(port, edge);
}

process::port_info_t
mutate_process
::_input_port_info(port_t const& port) const
{
  if (port == priv::INPUT_PORT_NAME)
  {
    return d->input_port_info;
  }

  return process::_input_port_info(port);
}

process::ports_t
mutate_process
::_input_ports() const
{
  ports_t ports;

  ports.push_back(priv::INPUT_PORT_NAME);

  return ports;
}

mutate_process::priv
::priv()
{
  port_flags_t mutate_required;

  mutate_required.insert(flag_required);
  mutate_required.insert(flag_input_mutable);

  input_port_info = port_info_t(new port_info(
    type_any,
    mutate_required,
    port_description_t("The port with the mutate flag set.")));
}

mutate_process::priv
::~priv()
{
}

}
