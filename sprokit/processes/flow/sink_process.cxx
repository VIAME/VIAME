// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "sink_process.h"

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>

/**
 * \file sink_process.cxx
 *
 * \brief Implementation of the sink process.
 */

namespace sprokit
{

class sink_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const sink_process::priv::port_input = port_t("sink");

sink_process
::sink_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    type_any,
    required,
    port_description_t("The data to ignore."));
}

sink_process
::~sink_process()
{
}

void
sink_process
::_step()
{
  (void)grab_from_port(priv::port_input);

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
