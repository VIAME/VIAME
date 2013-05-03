/*ckwg +5
 * Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "expect_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

/**
 * \file expect_process.cxx
 *
 * \brief Implementation of the expect process.
 */

namespace vistk
{

class expect_process::priv
{
  public:
    priv(std::string const& expect_, bool expect_key_);
    ~priv();

    std::string const expect;
    bool const expect_key;

    static config::key_t const config_tunable;
    static config::key_t const config_expect;
    static config::key_t const config_expect_key;
    static config::value_t const default_expect_key;
    static port_t const port_output;
};

config::key_t const expect_process::priv::config_tunable = config::key_t("tunable");
config::key_t const expect_process::priv::config_expect = config::key_t("expect");
config::key_t const expect_process::priv::config_expect_key = config::key_t("expect_key");
config::value_t const expect_process::priv::default_expect_key = config::value_t("false");
process::port_t const expect_process::priv::port_output = port_t("dummy");

expect_process
::expect_process(config_t const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_tunable,
    vistk::config::value_t(),
    vistk::config::description_t("A tunable value."),
    true);
  declare_configuration_key(
    priv::config_expect,
    vistk::config::value_t(),
    vistk::config::description_t("The expected value."));
  declare_configuration_key(
    priv::config_expect_key,
    priv::default_expect_key,
    vistk::config::description_t("Whether to expect a key or a value."));

  port_flags_t const none;

  declare_output_port(
    priv::port_output,
    type_none,
    none,
    port_description_t("A dummy port."));
}

expect_process
::~expect_process()
{
}

void
expect_process
::_configure()
{
  // Configure the process.
  {
    std::string const expect = config_value<std::string>(priv::config_expect);
    bool const expect_key = config_value<bool>(priv::config_expect_key);

    d.reset(new priv(expect, expect_key));
  }

  process::_configure();
}

void
expect_process
::_step()
{
  datum_t const dat = datum::empty_datum();

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

void
expect_process
::_reconfigure(vistk::config_t const& conf)
{
  if (d->expect_key)
  {
    if (!conf->has_value(d->expect))
    {
      static std::string const reason = "The expected key was not present on a reconfigure";

      throw invalid_configuration_exception(name(), reason);
    }
  }
  else
  {
    std::string const cur_value = config_value<std::string>(priv::config_tunable);

    if (cur_value != d->expect)
    {
      std::string const reason = "Did not get expected value: " + d->expect;

      throw invalid_configuration_value_exception(name(), priv::config_tunable, cur_value, reason);
    }
  }

  process::_reconfigure(conf);
}

expect_process::priv
::priv(std::string const& expect_, bool expect_key_)
  : expect(expect_)
  , expect_key(expect_key_)
{
}

expect_process::priv
::~priv()
{
}

}
