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

#include "const_number_process.h"

#include <vital/config/config_block.h>

/**
 * \file const_number_process.cxx
 *
 * \brief Implementation of the constant number process.
 */

namespace sprokit
{

class const_number_process::priv
{
  public:
    typedef int32_t number_t;

    priv(number_t v);
    ~priv();

    number_t const value;

    static kwiver::vital::config_block_key_t const config_value;
    static kwiver::vital::config_block_value_t const default_value;
    static port_t const port_output;
};

kwiver::vital::config_block_key_t const const_number_process::priv::config_value = kwiver::vital::config_block_key_t("value");
kwiver::vital::config_block_value_t const const_number_process::priv::default_value = kwiver::vital::config_block_value_t("0");
process::port_t const const_number_process::priv::port_output = port_t("number");

const_number_process
::const_number_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_value,
    priv::default_value,
    kwiver::vital::config_block_description_t("The value to start counting at."));

  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(
    priv::port_output,
    "integer",
    required,
    port_description_t("Where the numbers will be available."));
}

const_number_process
::~const_number_process()
{
}

void
const_number_process
::_configure()
{
  // Configure the process.
  {
    priv::number_t const value = config_value<priv::number_t>(priv::config_value);

    d.reset(new priv(value));
  }

  process::_configure();
}

void
const_number_process
::_step()
{
  push_to_port_as<priv::number_t>(priv::port_output, d->value);

  process::_step();
}

const_number_process::priv
::priv(number_t v)
  : value(v)
{
}

const_number_process::priv
::~priv()
{
}

}
