/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "take_number_process.h"

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/process_exception.h>

#include <boost/cstdint.hpp>
#include <boost/make_shared.hpp>

#include <string>

/**
 * \file take_number_process.cxx
 *
 * \brief Implementation of the number taking process.
 */

namespace vistk
{

class take_number_process::priv
{
  public:
    typedef int32_t number_t;

    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const take_number_process::priv::port_input = process::port_t("number");

take_number_process
::take_number_process(config_t const& config)
  : process(config)
{
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    "integer",
    required,
    port_description_t("Where numbers are read from.")));
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
