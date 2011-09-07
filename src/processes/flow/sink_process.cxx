/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "sink_process.h"

#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>

#include <boost/make_shared.hpp>

/**
 * \file sink_process.cxx
 *
 * \brief Implementation of the sink process.
 */

namespace vistk
{

class sink_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const sink_process::priv::port_input = process::port_t("sink");

sink_process
::sink_process(config_t const& config)
  : process(config)
{
  d.reset(new priv);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    type_any,
    required,
    port_description_t("The data to ignore.")));
}

sink_process
::~sink_process()
{
}

void
sink_process
::_step()
{
  edge_datum_t const input_dat = grab_from_port(priv::port_input);

  switch (input_dat.get<0>()->type())
  {
    case datum::complete:
      mark_as_complete();
      break;
    case datum::data:
    case datum::empty:
    case datum::error:
    case datum::invalid:
    default:
      break;
  }

  process::_step();
}

sink_process::priv
::priv()
{
}

sink_process::priv
::~priv()
{
}

}
