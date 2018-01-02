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

#include "multiplication_process.h"

/**
 * \file multiplication_process.cxx
 *
 * \brief Implementation of the multiplication process.
 */

namespace sprokit
{

class multiplication_process::priv
{
  public:
    typedef int32_t number_t;

    priv();
    ~priv();

    static port_t const port_factor1;
    static port_t const port_factor2;
    static port_t const port_output;
};

process::port_t const multiplication_process::priv::port_factor1 = port_t("factor1");
process::port_t const multiplication_process::priv::port_factor2 = port_t("factor2");
process::port_t const multiplication_process::priv::port_output = port_t("product");

multiplication_process
::multiplication_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_factor1,
    "integer",
    required,
    port_description_t("The first factor to multiply."));
  declare_input_port(
    priv::port_factor2,
    "integer",
    required,
    port_description_t("The second factor to multiply."));
  declare_output_port(
    priv::port_output,
    "integer",
    required,
    port_description_t("Where the product will be available."));
}

multiplication_process
::~multiplication_process()
{
}

void
multiplication_process
::_step()
{
  priv::number_t const factor1 = grab_from_port_as<priv::number_t>(priv::port_factor1);
  priv::number_t const factor2 = grab_from_port_as<priv::number_t>(priv::port_factor2);

  priv::number_t const product = factor1 * factor2;

  push_to_port_as<priv::number_t>(priv::port_output, product);

  process::_step();
}

multiplication_process::priv
::priv()
{
}

multiplication_process::priv
::~priv()
{
}

}
