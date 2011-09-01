/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "const_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>

#include <boost/make_shared.hpp>

namespace vistk
{

class const_process::priv
{
  public:
    priv();
    ~priv();

    edge_group_t output_edges;

    port_info_t output_port_info;

    static port_t const port_output;
};

process::port_t const const_process::priv::port_output = process::port_t("const");

const_process
::const_process(config_t const& config)
  : process(config)
{
  d.reset(new priv);

  port_flags_t const_required;

  const_required.insert(flag_required);
  const_required.insert(flag_output_const);

  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    type_none,
    const_required,
    port_description_t("The port with the const flag set.")));
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

  push_to_port(priv::port_output, edat);

  process::_step();
}

const_process::priv
::priv()
{
}

const_process::priv
::~priv()
{
}

}
