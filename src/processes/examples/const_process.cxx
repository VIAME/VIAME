/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "const_process.h"

#include <vistk/pipeline/datum.h>

#include <boost/make_shared.hpp>

/**
 * \file const_process.cxx
 *
 * \brief Implementation of the const process.
 */

namespace vistk
{

class const_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_output;
};

process::port_t const const_process::priv::port_output = process::port_t("const");

const_process
::const_process(config_t const& config)
  : process(config)
  , d(new priv)
{
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
  push_datum_to_port(priv::port_output, datum::empty_datum());

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
