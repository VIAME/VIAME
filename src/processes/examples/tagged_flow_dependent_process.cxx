/*ckwg +29
 * Copyright 2012 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
