// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "mutate_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file mutate_process.cxx
 *
 * \brief Implementation of the mutate process.
 */

namespace sprokit
{

class mutate_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const mutate_process::priv::port_input = port_t("mutate");

mutate_process
::mutate_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t mutate_required;

  mutate_required.insert(flag_required);
  mutate_required.insert(flag_input_mutable);

  declare_input_port(
    priv::port_input,
    type_any,
    mutate_required,
    port_description_t("The port with the mutate flag set."));
}

mutate_process
::~mutate_process()
{
}

void
mutate_process
::_step()
{
  (void)grab_from_port(priv::port_input);

  process::_step();
}

mutate_process::priv
::priv()
{
}

mutate_process::priv
::~priv()
{
}

}
