/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "timestamper_process.h"

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/edge.h>
#include <vistk/pipeline/process_exception.h>

#include <vistk/utilities/timestamp.h>

#include <boost/make_shared.hpp>

#include <string>

/**
 * \file timestamper_process.cxx
 *
 * \brief Implementation of the timestamper process.
 */

namespace vistk
{

class timestamper_process::priv
{
  public:
    typedef double frame_rate_t;

    priv(timestamp::frame_t start_frame, timestamp::time_t start_time, frame_rate_t rate);
    ~priv();

    timestamp::frame_t frame;
    timestamp::time_t time;
    frame_rate_t const time_diff;

    static config::key_t const config_start_frame;
    static config::key_t const config_start_time;
    static config::key_t const config_frame_rate;
    static config::value_t const default_start_frame;
    static config::value_t const default_start_time;
    static config::value_t const default_frame_rate;
    static port_t const port_output;
};

config::key_t const timestamper_process::priv::config_start_frame = config::key_t("start_frame");
config::key_t const timestamper_process::priv::config_start_time = config::key_t("start_time");
config::key_t const timestamper_process::priv::config_frame_rate = config::key_t("frame_rate");
config::value_t const timestamper_process::priv::default_start_frame = config::value_t("0");
config::value_t const timestamper_process::priv::default_start_time = config::value_t("0");
config::value_t const timestamper_process::priv::default_frame_rate = config::value_t("30");
process::port_t const timestamper_process::priv::port_output = port_t("timestamp");

timestamper_process
::timestamper_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(priv::config_start_frame, boost::make_shared<conf_info>(
    priv::default_start_frame,
    config::description_t("")));
  declare_configuration_key(priv::config_start_time, boost::make_shared<conf_info>(
    priv::default_start_time,
    config::description_t("")));
  declare_configuration_key(priv::config_frame_rate, boost::make_shared<conf_info>(
    priv::default_frame_rate,
    config::description_t("")));

  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(priv::port_output, boost::make_shared<port_info>(
    "timestamp",
    required,
    port_description_t("Where the timestamps will be available.")));
}

timestamper_process
::~timestamper_process()
{
}

void
timestamper_process
::_init()
{
  // Configure the process.
  {
    timestamp::frame_t const start_frame = config_value<timestamp::frame_t>(priv::config_start_frame);
    timestamp::time_t const start_time = config_value<timestamp::time_t>(priv::config_start_time);
    priv::frame_rate_t const rate = config_value<priv::frame_rate_t>(priv::config_frame_rate);

    d.reset(new priv(start_frame, start_time, rate));
  }

  if (d->time < 0)
  {
    static std::string const reason = "The start_time must be greater than 0";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_init();
}

void
timestamper_process
::_step()
{
  datum_t dat;

  timestamp ts = timestamp(d->frame, d->time);

  d->time += d->time_diff;

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

timestamper_process::priv
::priv(timestamp::frame_t start_frame, timestamp::time_t start_time, frame_rate_t rate)
  : frame(start_frame)
  , time(start_time)
  , time_diff(1.0 / rate)
{
}

timestamper_process::priv
::~priv()
{
}

}
