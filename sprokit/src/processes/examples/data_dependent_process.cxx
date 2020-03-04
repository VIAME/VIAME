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

#include "data_dependent_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file data_dependent_process.cxx
 *
 * \brief Implementation of the data dependent process.
 */

namespace sprokit
{

class data_dependent_process::priv
{
  public:
    priv(bool reject_, bool set_on_configure_);
    ~priv();

    bool const reject;
    bool const set_on_configure;
    bool configuring;
    bool type_set;

    static kwiver::vital::config_block_key_t const config_reject;
    static kwiver::vital::config_block_key_t const config_set_on_configure;
    static kwiver::vital::config_block_value_t const default_reject;
    static kwiver::vital::config_block_value_t const default_set_on_configure;
    static port_t const port_output;
};

kwiver::vital::config_block_key_t const data_dependent_process::priv::config_reject =
  kwiver::vital::config_block_key_t("reject");
kwiver::vital::config_block_key_t const data_dependent_process::priv::config_set_on_configure =
  kwiver::vital::config_block_key_t("set_on_configure");
kwiver::vital::config_block_value_t const data_dependent_process::priv::default_reject =
  kwiver::vital::config_block_value_t("false");
kwiver::vital::config_block_value_t const data_dependent_process::priv::default_set_on_configure =
  kwiver::vital::config_block_value_t("true");
process::port_t const data_dependent_process::priv::port_output = port_t("output");

data_dependent_process
::data_dependent_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_reject,
    priv::default_reject,
    kwiver::vital::config_block_description_t("Whether to reject type setting requests or not."));

  declare_configuration_key(
    priv::config_set_on_configure,
    priv::default_set_on_configure,
    kwiver::vital::config_block_description_t("Whether to set the type on configure or not."));

  bool const reject = config_value<bool>(priv::config_reject);
  bool const set_on_configure = config_value<bool>(priv::config_set_on_configure);

  d.reset(new priv(reject, set_on_configure));

  make_ports();
}

data_dependent_process
::~data_dependent_process()
{
}

// ----------------------------------------------------------------------------
void
data_dependent_process
::_configure()
{
  d->configuring = true;

  // If the type has not already been set and we are to set type at
  // config time.
  if (!d->type_set && d->set_on_configure)
  {
    set_output_port_type(priv::port_output, type_none);
  }

  d->configuring = false;

  process::_configure();
}

void
data_dependent_process
::_step()
{
  push_datum_to_port(priv::port_output, datum::empty_datum());

  process::_step();
}

void
data_dependent_process
::_reset()
{
  d->type_set = false;

  remove_output_port(priv::port_output);

  make_ports();

  process::_reset();
}

bool
data_dependent_process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  if (!d->configuring && d->reject)
  {
    return false;
  }

  d->type_set = process::_set_output_port_type(port, new_type);

  return d->type_set;
}

void
data_dependent_process
::make_ports()
{
  declare_output_port(
    priv::port_output,
    type_data_dependent,
    port_flags_t(),
    port_description_t("An output port with a data dependent type"));
}

data_dependent_process::priv
::priv(bool reject_, bool set_on_configure_)
  : reject(reject_)
  , set_on_configure(set_on_configure_)
  , configuring(false)
  , type_set(false)
{
}

data_dependent_process::priv
::~priv()
{
}

}
