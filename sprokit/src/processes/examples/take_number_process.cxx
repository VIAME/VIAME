// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "take_number_process.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/process_exception.h>

#include <string>

/**
 * \file take_number_process.cxx
 *
 * \brief Implementation of the number taking process.
 */

namespace sprokit
{

class take_number_process::priv
{
  public:
    typedef int32_t number_t;

    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const take_number_process::priv::port_input = port_t("number");

take_number_process
::take_number_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    "integer",
    required,
    port_description_t("Where numbers are read from."));
}

take_number_process
::~take_number_process()
{
}

void
take_number_process
::_step()
{
  (void)grab_from_port_as<priv::number_t>(priv::port_input);

  process::_step();
}

take_number_process::priv
::priv()
{
}

take_number_process::priv
::~priv()
{
}

}
