/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "data_dependent_process.h"

#include <vistk/pipeline/datum.h>

#include <boost/make_shared.hpp>

/**
 * \file data_dependent_process.cxx
 *
 * \brief Implementation of the data dependent process.
 */

namespace vistk
{

class data_dependent_process::priv
{
  public:
    priv(bool reject_);
    ~priv();

    bool const reject;
    bool initializing;
    bool type_set;

    static config::key_t const config_reject;
    static config::value_t const default_reject;
    static port_t const port_output;
};

config::key_t const data_dependent_process::priv::config_reject = config::key_t("reject");
config::value_t const data_dependent_process::priv::default_reject = config::value_t("false");
process::port_t const data_dependent_process::priv::port_output = process::port_t("output");

data_dependent_process
::data_dependent_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_reject, boost::make_shared<conf_info>(
    priv::default_reject,
    config::description_t("Whether to reject type setting requests or not.")));

  bool const reject = config_value<bool>(priv::config_reject);

  d.reset(new priv(reject));

  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    type_data_dependent,
    port_flags_t(),
    port_description_t("An output port with a data dependent type")));
}

data_dependent_process
::~data_dependent_process()
{
}

void
data_dependent_process
::_init()
{
  d->initializing = true;

  if (!d->type_set)
  {
    set_output_port_type(priv::port_output, type_none);
  }

  process::_init();
}

void
data_dependent_process
::_step()
{
  push_datum_to_port(priv::port_output, datum::empty_datum());

  process::_step();
}

bool
data_dependent_process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  if (!d->initializing && d->reject)
  {
    return false;
  }

  d->type_set = process::_set_output_port_type(port, new_type);

  return d->type_set;
}

data_dependent_process::priv
::priv(bool reject_)
  : reject(reject_)
  , initializing(false)
  , type_set(false)
{
}

data_dependent_process::priv
::~priv()
{
}

}
