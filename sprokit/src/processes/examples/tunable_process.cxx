// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "tunable_process.h"

#include <vital/config/config_block.h>
#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/process_exception.h>

#include <string>

/**
 * \file tunable_process.cxx
 *
 * \brief Implementation of the tunable process.
 */

namespace sprokit
{

class tunable_process::priv
{
  public:
    priv(std::string t, std::string nt);
    ~priv();

    std::string tunable;
    std::string const non_tunable;

    static kwiver::vital::config_block_key_t const config_tunable;
    static kwiver::vital::config_block_key_t const config_non_tunable;
    static port_t const port_tunable;
    static port_t const port_non_tunable;
};

kwiver::vital::config_block_key_t const tunable_process::priv::config_tunable = kwiver::vital::config_block_key_t("tunable");
kwiver::vital::config_block_key_t const tunable_process::priv::config_non_tunable = kwiver::vital::config_block_key_t("non_tunable");
process::port_t const tunable_process::priv::port_tunable = process::port_t("tunable");
process::port_t const tunable_process::priv::port_non_tunable = process::port_t("non_tunable");

tunable_process
::tunable_process(kwiver::vital::config_block_sptr const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_tunable,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("The tunable output."),
    true);
  declare_configuration_key(
    priv::config_non_tunable,
    kwiver::vital::config_block_value_t(),
    kwiver::vital::config_block_description_t("The non-tunable output."));

  port_flags_t const none;

  declare_output_port(
    priv::port_tunable,
    "string",
    none,
    port_description_t("The tunable output."));
  declare_output_port(
    priv::port_non_tunable,
    "string",
    none,
    port_description_t("The non-tunable output."));
}

tunable_process
::~tunable_process()
{
}

void
tunable_process
::_configure()
{
  // Configure the process.
  {
    std::string const tunable = config_value<std::string>(priv::config_tunable);
    std::string const non_tunable = config_value<std::string>(priv::config_non_tunable);

    d.reset(new priv(tunable, non_tunable));
  }

  process::_configure();
}

void
tunable_process
::_step()
{
  push_to_port_as<std::string>(priv::port_tunable, d->tunable);
  push_to_port_as<std::string>(priv::port_non_tunable, d->non_tunable);

  mark_process_as_complete();

  process::_step();
}

void
tunable_process
::_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  d->tunable = config_value<std::string>(priv::config_tunable);

  process::_reconfigure(conf);
}

tunable_process::priv
::priv(std::string t, std::string nt)
  : tunable(t)
  , non_tunable(nt)
{
}

tunable_process::priv
::~priv()
{
}

}
