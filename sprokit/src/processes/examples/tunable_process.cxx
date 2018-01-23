/*ckwg +29
 * Copyright 2013 by Kitware, Inc.
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

#include "tunable_process.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <string>

/**
 * \file tunable_process.cxx
 *
 * \brief Implementation of the tunable process.
 */

namespace sprokit
{

class tunable_process::priv
{
  public:
    priv(std::string t, std::string nt);
    ~priv();

    std::string tunable;
    std::string const non_tunable;

    static kwiver::vital::config_block_key_t const config_tunable;
    static kwiver::vital::config_block_key_t const config_non_tunable;
    static port_t const port_tunable;
    static port_t const port_non_tunable;
};

kwiver::vital::config_block_key_t const tunable_process::priv::config_tunable = kwiver::vital::config_block_key_t("tunable");
kwiver::vital::config_block_key_t const tunable_process::priv::config_non_tunable = kwiver::vital::config_block_key_t("non_tunable");
process::port_t const tunable_process::priv::port_tunable = process::port_t("tunable");
process::port_t const tunable_process::priv::port_non_tunable = process::port_t("non_tunable");

tunable_process
::tunable_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_tunable,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("The tunable output."),
    true);
  declare_configuration_key(
    priv::config_non_tunable,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("The non-tunable output."));

  port_flags_t const none;

  declare_output_port(
    priv::port_tunable,
    "string",
    none,
    port_description_t("The tunable output."));
  declare_output_port(
    priv::port_non_tunable,
    "string",
    none,
    port_description_t("The non-tunable output."));
}

tunable_process
::~tunable_process()
{
}

void
tunable_process
::_configure()
{
  // Configure the process.
  {
    std::string const tunable = config_value<std::string>(priv::config_tunable);
    std::string const non_tunable = config_value<std::string>(priv::config_non_tunable);

    d.reset(new priv(tunable, non_tunable));
  }

  process::_configure();
}

void
tunable_process
::_step()
{
  push_to_port_as<std::string>(priv::port_tunable, d->tunable);
  push_to_port_as<std::string>(priv::port_non_tunable, d->non_tunable);

  mark_process_as_complete();

  process::_step();
}

void
tunable_process
::_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  d->tunable = config_value<std::string>(priv::config_tunable);

  process::_reconfigure(conf);
}

tunable_process::priv
::priv(std::string t, std::string nt)
  : tunable(t)
  , non_tunable(nt)
{
}

tunable_process::priv
::~priv()
{
}

}
