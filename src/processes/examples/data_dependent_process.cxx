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
    priv(bool reject_, bool set_on_analysis_);
    ~priv();

    bool const reject;
    bool const set_on_analysis;
    bool analyzing;
    bool type_set;

    static config::key_t const config_reject;
    static config::key_t const config_set_on_analysis;
    static config::value_t const default_reject;
    static config::value_t const default_set_on_analysis;
    static port_t const port_output;
};

config::key_t const data_dependent_process::priv::config_reject = config::key_t("reject");
config::key_t const data_dependent_process::priv::config_set_on_analysis = config::key_t("set_on_analysis");
config::value_t const data_dependent_process::priv::default_reject = config::value_t("false");
config::value_t const data_dependent_process::priv::default_set_on_analysis = config::value_t("true");
process::port_t const data_dependent_process::priv::port_output = process::port_t("output");

data_dependent_process
::data_dependent_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_reject, boost::make_shared<conf_info>(
    priv::default_reject,
    config::description_t("Whether to reject type setting requests or not.")));
  declare_configuration_key(priv::config_set_on_analysis, boost::make_shared<conf_info>(
    priv::default_set_on_analysis,
    config::description_t("Whether to set the type on analysis or not.")));

  bool const reject = config_value<bool>(priv::config_reject);
  bool const set_on_analysis = config_value<bool>(priv::config_set_on_analysis);

  d.reset(new priv(reject, set_on_analysis));

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
::_analyze()
{
  d->analyzing = true;

  if (!d->type_set && d->set_on_analysis)
  {
    set_output_port_type(priv::port_output, type_none);
  }

  process::_analyze();
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
  if (!d->analyzing && d->reject)
  {
    return false;
  }

  d->type_set = process::_set_output_port_type(port, new_type);

  return d->type_set;
}

data_dependent_process::priv
::priv(bool reject_, bool set_on_analysis_)
  : reject(reject_)
  , set_on_analysis(set_on_analysis_)
  , analyzing(false)
  , type_set(false)
{
}

data_dependent_process::priv
::~priv()
{
}

}
