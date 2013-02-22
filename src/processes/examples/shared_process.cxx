/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "shared_process.h"

#include <vistk/pipeline/datum.h>

/**
 * \file shared_process.cxx
 *
 * \brief Implementation of the shared process.
 */

namespace vistk
{

class shared_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_output;
};

process::port_t const shared_process::priv::port_output = port_t("shared");

shared_process
::shared_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t shared_required;

  shared_required.insert(flag_required);
  shared_required.insert(flag_output_shared);

  declare_output_port(
    priv::port_output,
    type_none,
    shared_required,
    port_description_t("The port with the shared flag set."));
}

shared_process
::~shared_process()
{
}

void
shared_process
::_step()
{
  push_datum_to_port(priv::port_output, datum::empty_datum());

  process::_step();
}

shared_process::priv
::priv()
{
}

shared_process::priv
::~priv()
{
}

}
