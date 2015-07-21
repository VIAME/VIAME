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

#include "feedback_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file feedback_process.cxx
 *
 * \brief Implementation of the feedback process.
 */

namespace sprokit
{

class feedback_process::priv
{
  public:
    priv();
    ~priv();

    bool first;

    static port_t const port_input;
    static port_t const port_output;
    static port_type_t const type_custom_feedback;
};

process::port_t const feedback_process::priv::port_input = port_t("input");
process::port_t const feedback_process::priv::port_output = port_t("output");
process::port_type_t const feedback_process::priv::type_custom_feedback = port_type_t("__feedback");

feedback_process
::feedback_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t flags;

  flags.insert(flag_required);

  declare_output_port(
    priv::port_output,
    priv::type_custom_feedback,
    flags,
    port_description_t("A port which outputs data for this process\' input."));

  flags.insert(flag_input_nodep);

  declare_input_port(
    priv::port_input,
    priv::type_custom_feedback,
    flags,
    port_description_t("A port which accepts this process\' output."));
}

feedback_process
::~feedback_process()
{
}

void
feedback_process
::_flush()
{
  d->first = true;

  process::_flush();
}

void
feedback_process
::_step()
{
  if (d->first)
  {
    push_datum_to_port(priv::port_output, datum::empty_datum());

    d->first = false;
  }
  else
  {
    push_datum_to_port(priv::port_output, grab_datum_from_port(priv::port_input));
  }

  process::_step();
}

feedback_process::priv
::priv()
  : first(true)
{
}

feedback_process::priv
::~priv()
{
}

}
