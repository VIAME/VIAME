// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "tagged_flow_dependent_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file tagged_flow_dependent_process.cxx
 *
 * \brief Implementation of the flow dependent process.
 */

namespace sprokit
{

class tagged_flow_dependent_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t tag_t;

    static port_t const port_untagged_input;
    static port_t const port_tagged_input;
    static port_t const port_untagged_output;
    static port_t const port_tagged_output;
};

process::port_t const tagged_flow_dependent_process::priv::port_untagged_input = port_t("untagged_input");
process::port_t const tagged_flow_dependent_process::priv::port_tagged_input = port_t("tagged_input");
process::port_t const tagged_flow_dependent_process::priv::port_untagged_output = port_t("untagged_output");
process::port_t const tagged_flow_dependent_process::priv::port_tagged_output = port_t("tagged_output");

tagged_flow_dependent_process
::tagged_flow_dependent_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  make_ports();
}

tagged_flow_dependent_process
::~tagged_flow_dependent_process()
{
}

void
tagged_flow_dependent_process
::_reset()
{
  remove_input_port(priv::port_untagged_input);
  remove_input_port(priv::port_tagged_input);
  remove_output_port(priv::port_untagged_output);
  remove_output_port(priv::port_tagged_output);

  make_ports();

  process::_reset();
}

void
tagged_flow_dependent_process
::_step()
{
  push_datum_to_port(priv::port_untagged_output, grab_datum_from_port(priv::port_untagged_input));
  push_datum_to_port(priv::port_tagged_output, grab_datum_from_port(priv::port_tagged_input));

  process::_step();
}

void
tagged_flow_dependent_process
::make_ports()
{
  priv::tag_t const tag = "tag";

  declare_input_port(priv::port_untagged_input,
    type_flow_dependent,
    port_flags_t(),
    port_description_t("An untagged input port with a flow dependent type."));
  declare_input_port(priv::port_tagged_input,
    type_flow_dependent + tag,
    port_flags_t(),
    port_description_t("A tagged input port with a flow dependent type."));

  declare_output_port(priv::port_untagged_output,
    type_flow_dependent,
    port_flags_t(),
    port_description_t("An untagged output port with a flow dependent type"));
  declare_output_port(priv::port_tagged_output,
    type_flow_dependent + tag,
    port_flags_t(),
    port_description_t("A tagged output port with a flow dependent type"));
}

tagged_flow_dependent_process::priv
::priv()
{
}

tagged_flow_dependent_process::priv
::~priv()
{
}

}
