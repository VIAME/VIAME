// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pass_process.h"

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>

/**
 * \file pass_process.cxx
 *
 * \brief Implementation of the pass process.
 */

namespace sprokit
{

class pass_process::priv
{
  public:
    priv();
    ~priv();

    typedef port_t tag_t;

    static port_t const port_input;
    static port_t const port_output;
    static tag_t const tag;
};

process::port_t const pass_process::priv::port_input = port_t("pass");
process::port_t const pass_process::priv::port_output = port_t("pass");
pass_process::priv::tag_t const pass_process::priv::tag = tag_t("pass");

pass_process
::pass_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv)
{
  // Special cases are handled by the process.
  set_data_checking_level(check_sync);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    type_flow_dependent + priv::tag,
    required,
    port_description_t("The datum to pass."));

  declare_output_port(
    priv::port_output,
    type_flow_dependent + priv::tag,
    required,
    port_description_t("The passed datum."));
}

pass_process
::~pass_process()
{
}

void
pass_process
::_step()
{
  datum_t const dat = grab_datum_from_port(priv::port_input);
  bool const complete = (dat->type() == datum::complete);

  push_datum_to_port(priv::port_output, dat);

  if (complete)
  {
    mark_process_as_complete();
  }

  process::_step();
}

pass_process::priv
::priv()
{
}

pass_process::priv
::~priv()
{
}

}
