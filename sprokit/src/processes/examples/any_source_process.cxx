/*ckwg +29
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
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

#include "any_source_process.h"

#include <vital/config/config_block.h>

/**
 * \file any_source_process.cxx
 *
 * \brief Implementation of the constant number process.
 */

namespace sprokit
{

class any_source_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_output;
};

process::port_t const any_source_process::priv::port_output = port_t("data");

any_source_process
::any_source_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(
    priv::port_output,
    type_any,
    required,
    port_description_t("The data."));
}

any_source_process
::~any_source_process()
{
}

void
any_source_process
::_step()
{
  // We can't create "arbitrary" data, so we make empties.
  datum_t const dat = datum::empty_datum();
  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

any_source_process::priv
::priv()
{
}

any_source_process::priv
::~priv()
{
}

}
