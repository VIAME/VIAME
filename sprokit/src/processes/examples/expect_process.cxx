/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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

#include "expect_process.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

/**
 * \file expect_process.cxx
 *
 * \brief Implementation of the expect process.
 */

namespace sprokit
{

class expect_process::priv
{
  public:
    priv(std::string const& expect_, bool expect_key_);
    ~priv();

    std::string const expect;
    bool const expect_key;

    static kwiver::vital::config_block_key_t const config_tunable;
    static kwiver::vital::config_block_key_t const config_expect;
    static kwiver::vital::config_block_key_t const config_expect_key;
    static kwiver::vital::config_block_value_t const default_expect_key;
    static port_t const port_output;
};

kwiver::vital::config_block_key_t const expect_process::priv::config_tunable = kwiver::vital::config_block_key_t("tunable");
kwiver::vital::config_block_key_t const expect_process::priv::config_expect = kwiver::vital::config_block_key_t("expect");
kwiver::vital::config_block_key_t const expect_process::priv::config_expect_key = kwiver::vital::config_block_key_t("expect_key");
kwiver::vital::config_block_value_t const expect_process::priv::default_expect_key = kwiver::vital::config_block_value_t("false");
process::port_t const expect_process::priv::port_output = port_t("dummy");

expect_process
::expect_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_tunable,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("A tunable value."),
    true);
  declare_configuration_key(
    priv::config_expect,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("The expected value."));
  declare_configuration_key(
    priv::config_expect_key,
    priv::default_expect_key,
    kwiver::vital::config_block_description_t("Whether to expect a key or a value."));

  port_flags_t const none;

  declare_output_port(
    priv::port_output,
    type_none,
    none,
    port_description_t("A dummy port."));
}

expect_process
::~expect_process()
{
}

void
expect_process
::_configure()
{
  // Configure the process.
  {
    std::string const expect = config_value<std::string>(priv::config_expect);
    bool const expect_key = config_value<bool>(priv::config_expect_key);

    d.reset(new priv(expect, expect_key));
  }

  process::_configure();
}

void
expect_process
::_step()
{
  datum_t const dat = datum::empty_datum();

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

void
expect_process
::_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  if (d->expect_key)
  {
    // check if key exists
    if (!conf->has_value(d->expect))
    {
      static std::string const reason = "The expected key was not present on a reconfigure";

      VITAL_THROW( invalid_configuration_exception,
                   name(), reason);
    }
  }
  else // expect new value
  {
    std::string const cur_value = config_value<std::string>(priv::config_tunable);
    if (cur_value != d->expect)
    {
      std::string const reason = "Did not get expected value: " + d->expect;

      VITAL_THROW( invalid_configuration_value_exception,
                   name(), priv::config_tunable, cur_value, reason);
    }
  }

  process::_reconfigure(conf); // pass to base class
}

expect_process::priv
::priv(std::string const& expect_, bool expect_key_)
  : expect(expect_)
  , expect_key(expect_key_)
{
}

expect_process::priv
::~priv()
{
}

}
