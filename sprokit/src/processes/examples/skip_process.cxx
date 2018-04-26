/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include "skip_process.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <string>

/**
 * \file skip_process.cxx
 *
 * \brief Implementation of the skip data process.
 */

namespace sprokit
{

class skip_process::priv
{
  public:
    priv(frequency_component_t s, frequency_component_t o);
    ~priv();

    typedef port_t tag_t;

    frequency_component_t const skip;
    frequency_component_t const offset;

    static kwiver::vital::config_block_key_t const config_skip;
    static kwiver::vital::config_block_value_t const default_skip;
    static kwiver::vital::config_block_key_t const config_offset;
    static kwiver::vital::config_block_value_t const default_offset;
    static port_t const port_input;
    static port_t const port_output;
};

kwiver::vital::config_block_key_t const skip_process::priv::config_skip = kwiver::vital::config_block_key_t("skip");
kwiver::vital::config_block_value_t const skip_process::priv::default_skip = kwiver::vital::config_block_key_t("1");
kwiver::vital::config_block_key_t const skip_process::priv::config_offset = kwiver::vital::config_block_key_t("offset");
kwiver::vital::config_block_value_t const skip_process::priv::default_offset = kwiver::vital::config_block_key_t("0");
process::port_t const skip_process::priv::port_input = port_t("input");
process::port_t const skip_process::priv::port_output = port_t("output");

skip_process
::skip_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv(1, 0))
{
  declare_configuration_key(
    priv::config_skip,
    priv::default_skip,
    kwiver::vital::config_block_description_t("The number of inputs to skip for each output."));
  declare_configuration_key(
    priv::config_offset,
    priv::default_offset,
    kwiver::vital::config_block_description_t("The offset from the first datum to use as the output."));

  port_flags_t required;

  required.insert(flag_required);

  priv::tag_t const tag = priv::tag_t("tag");

  declare_input_port(
    priv::port_input,
    type_flow_dependent + tag,
    required,
    port_description_t("A stream with extra data at regular intervals."));

  declare_output_port(
    priv::port_output,
    type_flow_dependent + tag,
    required,
    port_description_t("The input stream sampled at regular intervals."));
}

skip_process
::~skip_process()
{
}

void
skip_process
::_configure()
{
  // Configure the process.
  {
    frequency_component_t const skip = config_value<frequency_component_t>(priv::config_skip);
    frequency_component_t const offset = config_value<frequency_component_t>(priv::config_offset);

    d.reset(new priv(skip, offset));
  }

  if (d->skip <= d->offset)
  {
    static std::string const reason = "The offset must be less than the skip count";

    VITAL_THROW( invalid_configuration_exception,
                 name(), reason);
  }

  set_input_port_frequency(priv::port_input, 1 + d->skip);

  process::_configure();
}

void
skip_process
::_step()
{
  for (frequency_component_t i = 0; i < d->skip; ++i)
  {
    datum_t const dat = grab_datum_from_port(priv::port_input);

    if (i == d->offset)
    {
      push_datum_to_port(priv::port_output, dat);
    }
  }

  process::_step();
}

skip_process::priv
::priv(frequency_component_t s, frequency_component_t o)
  : skip(s)
  , offset(o)
{
}

skip_process::priv
::~priv()
{
}

}
