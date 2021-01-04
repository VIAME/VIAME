// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
