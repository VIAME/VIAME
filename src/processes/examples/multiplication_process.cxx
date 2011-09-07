/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "multiplication_process.h"

#include <vistk/pipeline_types/basic_types.h>

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>

#include <boost/cstdint.hpp>
#include <boost/make_shared.hpp>

/**
 * \file multiplication_process.cxx
 *
 * \brief Implementation of the multiplication process.
 */

namespace vistk
{

class multiplication_process::priv
{
  public:
    typedef uint32_t number_t;

    priv();
    ~priv();

    static port_t const port_factor1;
    static port_t const port_factor2;
    static port_t const port_output;
};

process::port_t const multiplication_process::priv::port_factor1 = process::port_t("factor1");
process::port_t const multiplication_process::priv::port_factor2 = process::port_t("factor2");
process::port_t const multiplication_process::priv::port_output = process::port_t("product");

multiplication_process
::multiplication_process(config_t const& config)
  : process(config)
{
  d.reset(new priv);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_factor1, boost::make_shared<port_info>(
    basic_types::t_integer,
    required,
    port_description_t("The first factor to multiply.")));
  declare_input_port(priv::port_factor2, boost::make_shared<port_info>(
    basic_types::t_integer,
    required,
    port_description_t("The second factor to multiply.")));
  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    basic_types::t_integer,
    required,
    port_description_t("Where the product will be available.")));
}

multiplication_process
::~multiplication_process()
{
}

void
multiplication_process
::_step()
{
  edge_datum_t const factor1_dat = grab_from_port(priv::port_factor1);
  edge_datum_t const factor2_dat = grab_from_port(priv::port_factor2);

  edge_data_t input_dats;

  input_dats.push_back(factor1_dat);
  input_dats.push_back(factor2_dat);

  data_info_t const info = edge_data_info(input_dats);

  datum_t dat;
  stamp_t st;

  st = factor1_dat.get<1>();

  switch (info->max_status)
  {
    case datum::data:
      if (!info->same_color)
      {
        st = heartbeat_stamp();

        dat = datum::error_datum("The input edges are not colored the same.");
      }
      else if (!info->in_sync)
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
    case datum::empty:
      dat = datum::empty_datum();
      break;
    case datum::complete:
      mark_as_complete();
      dat = datum::complete_datum();
      break;
    case datum::error:
      dat = datum::error_datum("Error on the input edges.");
      break;
    case datum::invalid:
    default:
      dat = datum::error_datum("Unrecognized datum type.");
      break;
  }

  edge_datum_t const edat = edge_datum_t(dat, st);

  push_to_port(priv::port_output, edat);

  process::_step();
}

multiplication_process::priv
::priv()
{
}

multiplication_process::priv
::~priv()
{
}

}
