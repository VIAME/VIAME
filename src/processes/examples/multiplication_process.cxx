/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "multiplication_process.h"

#include <vistk/pipeline_types/port_types.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <algorithm>

namespace vistk
{

class multiplication_process::priv
{
  public:
    typedef uint32_t number_t;

    priv();
    ~priv();

    edge_ref_t input_edge_factor1;
    edge_ref_t input_edge_factor2;

    edge_group_t output_edges;

    port_info_t factor1_port_info;
    port_info_t factor2_port_info;
    port_info_t output_port_info;

    static port_t const INPUT_FACTOR1_PORT_NAME;
    static port_t const INPUT_FACTOR2_PORT_NAME;
    static port_t const OUTPUT_PORT_NAME;
};

process::port_t const multiplication_process::priv::INPUT_FACTOR1_PORT_NAME = process::port_t("factor1");
process::port_t const multiplication_process::priv::INPUT_FACTOR2_PORT_NAME = process::port_t("factor2");
process::port_t const multiplication_process::priv::OUTPUT_PORT_NAME = process::port_t("product");

multiplication_process
::multiplication_process(config_t const& config)
  : process(config)
{
  d = boost::shared_ptr<priv>(new priv);
}

multiplication_process
::~multiplication_process()
{
}

void
multiplication_process
::_step()
{
  datum_t dat;
  stamp_t st;

  edge_datum_t const factor1_dat = grab_from_edge_ref(d->input_edge_factor1);
  edge_datum_t const factor2_dat = grab_from_edge_ref(d->input_edge_factor2);

  edge_data_t input_dats;

  input_dats.push_back(factor1_dat);
  input_dats.push_back(factor2_dat);

  st = factor1_dat.get<1>();

  datum::datum_type_t const max_type = max_status(input_dats);

  switch (max_type)
  {
    case datum::DATUM_DATA:
      if (!same_colored_data(input_dats))
      {
        st = heartbeat_stamp();

        dat = datum::error_datum("The input edges are not colored the same.");
      }
      else if (!syncd_data(input_dats))
      {
        st = heartbeat_stamp();

        dat = datum::error_datum("The input edges are not synchronized.");
      }
      else
      {
        priv::number_t const factor1 = factor1_dat.get<0>()->get_datum<priv::number_t>();
        priv::number_t const factor2 = factor2_dat.get<0>()->get_datum<priv::number_t>();

        priv::number_t const product = factor1 * factor2;

        dat = datum::new_datum(product);
      }
      break;
    case datum::DATUM_EMPTY:
      dat = datum::empty_datum();
      break;
    case datum::DATUM_COMPLETE:
      mark_as_complete();
      dat = datum::complete_datum();
      break;
    case datum::DATUM_ERROR:
      dat = datum::error_datum("Error on the input edges.");
      break;
    case datum::DATUM_INVALID:
    default:
      dat = datum::error_datum("Unrecognized datum type.");
      break;
  }

  edge_datum_t const edat = edge_datum_t(dat, st);

  push_to_edges(d->output_edges, edat);

  process::_step();
}

void
multiplication_process
::_connect_input_port(port_t const& port, edge_ref_t edge)
{
  if (port == priv::INPUT_FACTOR1_PORT_NAME)
  {
    if (d->input_edge_factor1.use_count())
    {
      throw port_reconnect_exception(name(), port);
    }

    d->input_edge_factor1 = edge;

    return;
  }
  if (port == priv::INPUT_FACTOR2_PORT_NAME)
  {
    if (d->input_edge_factor2.use_count())
    {
      throw port_reconnect_exception(name(), port);
    }

    d->input_edge_factor2 = edge;

    return;
  }

  process::_connect_input_port(port, edge);
}

process::port_info_t
multiplication_process
::_input_port_info(port_t const& port) const
{
  if (port == priv::INPUT_FACTOR1_PORT_NAME)
  {
    return d->factor1_port_info;
  }
  if (port == priv::INPUT_FACTOR2_PORT_NAME)
  {
    return d->factor2_port_info;
  }

  return process::_input_port_info(port);
}

void
multiplication_process
::_connect_output_port(port_t const& port, edge_ref_t edge)
{
  if (port == priv::OUTPUT_PORT_NAME)
  {
    d->output_edges.push_back(edge);

    return;
  }

  process::_connect_output_port(port, edge);
}

process::port_info_t
multiplication_process
::_output_port_info(port_t const& port) const
{
  if (port == priv::OUTPUT_PORT_NAME)
  {
    return d->output_port_info;
  }

  return process::_output_port_info(port);
}

process::ports_t
multiplication_process
::_input_ports() const
{
  ports_t ports;

  ports.push_back(priv::INPUT_FACTOR1_PORT_NAME);
  ports.push_back(priv::INPUT_FACTOR2_PORT_NAME);

  return ports;
}

process::ports_t
multiplication_process
::_output_ports() const
{
  ports_t ports;

  ports.push_back(priv::OUTPUT_PORT_NAME);

  return ports;
}

multiplication_process::priv
::priv()
{
  port_flags_t required;

  required.insert(flag_required);

  factor1_port_info = port_info_t(new port_info(
    port_types::t_integer,
    required,
    port_description_t("The first factor to multiply.")));
  factor2_port_info = port_info_t(new port_info(
    port_types::t_integer,
    required,
    port_description_t("The second factor to multiply.")));
  output_port_info = port_info_t(new port_info(
    port_types::t_integer,
    required,
    port_description_t("Where the product will be available.")));
}

multiplication_process::priv
::~priv()
{
}

}
