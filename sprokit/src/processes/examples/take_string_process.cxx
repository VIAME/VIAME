// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
