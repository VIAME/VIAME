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

#include "flow_dependent_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file flow_dependent_process.cxx
 *
 * \brief Implementation of the flow dependent process.
 */

namespace sprokit
{

class flow_dependent_process::priv
{
  public:
    priv(bool reject_);
    ~priv();

    bool const reject;

    typedef port_t tag_t;

    static config::key_t const config_reject;
    static config::value_t const default_reject;
    static port_t const port_input;
    static port_t const port_output;
};

config::key_t const flow_dependent_process::priv::config_reject = config::key_t("reject");
config::value_t const flow_dependent_process::priv::default_reject = config::value_t("false");
process::port_t const flow_dependent_process::priv::port_input = port_t("input");
process::port_t const flow_dependent_process::priv::port_output = port_t("output");

flow_dependent_process
::flow_dependent_process(config_t const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_reject,
    priv::default_reject,
    config::description_t("Whether to reject type setting requests or not."));

  bool const reject = config_value<bool>(priv::config_reject);

  d.reset(new priv(reject));

  make_ports();
}

flow_dependent_process
::~flow_dependent_process()
{
}

void
flow_dependent_process
::_reset()
{
  remove_input_port(priv::port_input);
  remove_input_port(priv::port_output);

  make_ports();

  process::_reset();
}

void
flow_dependent_process
::_step()
{
  push_datum_to_port(priv::port_output, grab_datum_from_port(priv::port_input));

  process::_step();
}

bool
flow_dependent_process
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  if (d->reject)
  {
    return false;
  }

  return process::_set_input_port_type(port, new_type);
}

bool
flow_dependent_process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  if (d->reject)
  {
    return false;
  }

  return process::_set_output_port_type(port, new_type);
}

void
flow_dependent_process
::make_ports()
{
  priv::tag_t const tag = priv::tag_t("tag");

  declare_input_port(
    priv::port_input,
    type_flow_dependent + tag,
    port_flags_t(),
    port_description_t("An input port with a flow dependent type."));

  declare_output_port(
    priv::port_output,
    type_flow_dependent + tag,
    port_flags_t(),
    port_description_t("An output port with a flow dependent type"));
}

flow_dependent_process::priv
::priv(bool reject_)
  : reject(reject_)
{
}

flow_dependent_process::priv
::~priv()
{
}

}
