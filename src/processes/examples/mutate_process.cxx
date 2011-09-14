/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "mutate_process.h"

#include <vistk/pipeline/datum.h>

#include <boost/make_shared.hpp>

/**
 * \file mutate_process.cxx
 *
 * \brief Implementation of the mutate process.
 */

namespace vistk
{

class mutate_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
};

process::port_t const mutate_process::priv::port_input = process::port_t("mutate");

mutate_process
::mutate_process(config_t const& config)
  : process(config)
{
  d.reset(new priv);

  port_flags_t mutate_required;

  mutate_required.insert(flag_required);
  mutate_required.insert(flag_input_mutable);

  declare_input_port(priv::port_input, boost::make_shared<port_info>(
    type_any,
    mutate_required,
    port_description_t("The port with the mutate flag set.")));
}

mutate_process
::~mutate_process()
{
}

void
mutate_process
::_step()
{
  (void)grab_from_port(priv::port_input);

  process::_step();
}

mutate_process::priv
::priv()
{
}

mutate_process::priv
::~priv()
{
}

}
