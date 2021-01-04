// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
