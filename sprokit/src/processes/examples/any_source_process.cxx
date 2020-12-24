// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
