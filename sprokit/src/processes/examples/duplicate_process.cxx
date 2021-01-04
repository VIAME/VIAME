// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "duplicate_process.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <string>

/**
 * \file duplicate_process.cxx
 *
 * \brief Implementation of the duplicate process.
 */

namespace sprokit
{

class duplicate_process::priv
{
  public:
    priv(frequency_component_t c);
    ~priv();

    typedef port_t tag_t;

    frequency_component_t const copies;

    static kwiver::vital::config_block_key_t const config_copies;
    static kwiver::vital::config_block_value_t const default_copies;
    static port_t const port_input;
    static port_t const port_duplicate;
};

kwiver::vital::config_block_key_t const duplicate_process::priv::config_copies = kwiver::vital::config_block_key_t("copies");
kwiver::vital::config_block_value_t const duplicate_process::priv::default_copies = kwiver::vital::config_block_key_t("1");
process::port_t const duplicate_process::priv::port_input = port_t("input");
process::port_t const duplicate_process::priv::port_duplicate = port_t("duplicate");

duplicate_process
::duplicate_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d(new priv(1))
{
  declare_configuration_key(
    priv::config_copies,
    priv::default_copies,
    kwiver::vital::config_block_description_t("The number of copies to make of each input."));

  port_flags_t required;
  port_flags_t required_shared;

  required.insert(flag_required);
  required_shared.insert(flag_required);
  required_shared.insert(flag_output_shared);

  priv::tag_t const tag = priv::tag_t("tag");

  declare_input_port(
    priv::port_input,
    type_flow_dependent + tag,
    required,
    port_description_t("Arbitrary input data."));

  declare_output_port(
    priv::port_duplicate,
    type_flow_dependent + tag,
    required_shared,
    port_description_t("Duplicated input data."));
}

duplicate_process
::~duplicate_process()
{
}

void
duplicate_process
::_configure()
{
  // Configure the process.
  {
    frequency_component_t const copies = config_value<frequency_component_t>(priv::config_copies);

    d.reset(new priv(copies));
  }

  set_output_port_frequency(priv::port_duplicate, 1 + d->copies);

  process::_configure();
}

void
duplicate_process
::_step()
{
  datum_t const dat = grab_datum_from_port(priv::port_input);
  push_datum_to_port(priv::port_duplicate, dat);

  for (frequency_component_t i = 0; i < d->copies; ++i)
  {
    push_datum_to_port(priv::port_duplicate, dat);
  }

  process::_step();
}

duplicate_process::priv
::priv(frequency_component_t c)
  : copies(c)
{
}

duplicate_process::priv
::~priv()
{
}

}
