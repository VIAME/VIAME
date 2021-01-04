// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "const_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file const_process.cxx
 *
 * \brief Implementation of the const process.
 */

namespace sprokit
{

class const_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_output;
};

process::port_t const const_process::priv::port_output = port_t("const");

const_process
::const_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t const_required;

  const_required.insert(flag_required);
  const_required.insert(flag_output_const);

  declare_output_port(
    priv::port_output,
    type_none,
    const_required,
    port_description_t("The port with the const flag set."));
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
