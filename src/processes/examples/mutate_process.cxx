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

#include "mutate_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file mutate_process.cxx
 *
 * \brief Implementation of the mutate process.
 */

namespace sprokit
{

class mutate_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const mutate_process::priv::port_input = port_t("mutate");

mutate_process
::mutate_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t mutate_required;

  mutate_required.insert(flag_required);
  mutate_required.insert(flag_input_mutable);

  declare_input_port(
    priv::port_input,
    type_any,
    mutate_required,
    port_description_t("The port with the mutate flag set."));
}

mutate_process
::~mutate_process()
{
}

void
mutate_process
::_step()
{
  (void)grab_from_port(priv::port_input);

  process::_step();
}

mutate_process::priv
::priv()
{
}

mutate_process::priv
::~priv()
{
}

}
