/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "take_string_process.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_exception.h>

#include <string>

/**
 * \file take_string_process.cxx
 *
 * \brief Implementation of the string taking process.
 */

namespace vistk
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
::take_string_process(config_t const& config)
  : process(config)
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
