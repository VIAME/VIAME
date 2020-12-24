// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "flow_dependent_process.h"

#include <sprokit/pipeline/datum.h>

/**
 * \file flow_dependent_process.cxx
 *
 * \brief Implementation of the flow dependent process.
 */

namespace sprokit
{

class flow_dependent_process::priv
{
  public:
    priv(bool reject_);
    ~priv();

    bool const reject;

    typedef port_t tag_t;

    static kwiver::vital::config_block_key_t const config_reject;
    static kwiver::vital::config_block_value_t const default_reject;
    static port_t const port_input;
    static port_t const port_output;
};

kwiver::vital::config_block_key_t const flow_dependent_process::priv::config_reject = kwiver::vital::config_block_key_t("reject");
kwiver::vital::config_block_value_t const flow_dependent_process::priv::default_reject = kwiver::vital::config_block_value_t("false");
process::port_t const flow_dependent_process::priv::port_input = port_t("input");
process::port_t const flow_dependent_process::priv::port_output = port_t("output");

flow_dependent_process
::flow_dependent_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_reject,
    priv::default_reject,
    kwiver::vital::config_block_description_t("Whether to reject type setting requests or not."));

  bool const reject = config_value<bool>(priv::config_reject);

  d.reset(new priv(reject));

  make_ports();
}

flow_dependent_process
::~flow_dependent_process()
{
}

void
flow_dependent_process
::_reset()
{
  remove_input_port(priv::port_input);
  remove_input_port(priv::port_output);

  make_ports();

  process::_reset();
}

void
flow_dependent_process
::_step()
{
  push_datum_to_port(priv::port_output, grab_datum_from_port(priv::port_input));

  process::_step();
}

bool
flow_dependent_process
::_set_input_port_type(port_t const& port, port_type_t const& new_type)
{
  if (d->reject)
  {
    return false;
  }

  return process::_set_input_port_type(port, new_type);
}

bool
flow_dependent_process
::_set_output_port_type(port_t const& port, port_type_t const& new_type)
{
  if (d->reject)
  {
    return false;
  }

  return process::_set_output_port_type(port, new_type);
}

void
flow_dependent_process
::make_ports()
{
  priv::tag_t const tag = priv::tag_t("tag");

  declare_input_port(
    priv::port_input,
    type_flow_dependent + tag,
    port_flags_t(),
    port_description_t("An input port with a flow dependent type."));

  declare_output_port(
    priv::port_output,
    type_flow_dependent + tag,
    port_flags_t(),
    port_description_t("An output port with a flow dependent type"));
}

flow_dependent_process::priv
::priv(bool reject_)
  : reject(reject_)
{
}

flow_dependent_process::priv
::~priv()
{
}

}
