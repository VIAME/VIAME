// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "shared_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file shared_process.cxx
 *
 * \brief Implementation of the shared process.
 */

namespace sprokit
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
::shared_process(kwiver::vital::config_block_sptr const& config)
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
