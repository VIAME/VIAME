// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
