/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#include "take_string_process.h"

#include <sprokit/pipeline_util/path.h>

#include <vital/config/config_block.h>
#include <sprokit/pipeline/process_exception.h>

#include <string>

/**
 * \file take_string_process.cxx
 *
 * \brief Implementation of the string taking process.
 */

namespace sprokit
{

class take_string_process::priv
{
  public:
    typedef std::string string_t;

    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const take_string_process::priv::port_input = port_t("string");

take_string_process
::take_string_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    "string",
    required,
    port_description_t("Where strings are read from."));
}

take_string_process
::~take_string_process()
{
}

void
take_string_process
::_step()
{
  (void)grab_from_port_as<priv::string_t>(priv::port_input);

  process::_step();
}

take_string_process::priv
::priv()
{
}

take_string_process::priv
::~priv()
{
}

}
