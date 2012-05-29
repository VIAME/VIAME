/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "feedback_process.h"

#include <vistk/pipeline/datum.h>

#include <boost/make_shared.hpp>

/**
 * \file feedback_process.cxx
 *
 * \brief Implementation of the feedback process.
 */

namespace vistk
{

class feedback_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
    static port_t const port_output;
    static port_type_t const type_custom_feedback;
};

process::port_t const feedback_process::priv::port_input = process::port_t("input");
process::port_t const feedback_process::priv::port_output = process::port_t("output");
process::port_type_t const feedback_process::priv::type_custom_feedback = process::port_type_t("__feedback");

feedback_process
::feedback_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t flags;

  flags.insert(flag_required);

  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    priv::type_custom_feedback,
    flags,
    port_description_t("A port which outputs data for this process\' input.")));

  flags.insert(flag_input_nodep);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    priv::type_custom_feedback,
    flags,
    port_description_t("A port which accepts this process\' output.")));
}

feedback_process
::~feedback_process()
{
}

void
feedback_process
::_step()
{
  push_datum_to_port(priv::port_output, grab_datum_from_port(priv::port_input));

  process::_step();
}

feedback_process::priv
::priv()
{
}

feedback_process::priv
::~priv()
{
}

}
