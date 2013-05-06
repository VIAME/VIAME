/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "providers.h"

#include "path.h"
#include "pipe_bakery_exception.h"

#include <sprokit/pipeline/config.h>
#include <sprokit/pipeline/utils.h>

#include <boost/filesystem/operations.hpp>
#include <boost/system/error_code.hpp>
#include <boost/thread/thread.hpp>
#include <boost/lexical_cast.hpp>

#if defined(_WIN32) || defined(_WIN64)
#include <process.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

/**
 * \file providers.cxx
 *
 * \brief Implementation of configuration providers.
 */

namespace sprokit
{

provider
::provider()
{
}

provider
::~provider()
{
}

config_provider
::config_provider(config_t const conf)
  : m_config(conf)
{
}

config_provider
::~config_provider()
{
}

config::value_t
config_provider
::operator () (config::value_t const& index) const
{
  return m_config->get_value<config::value_t>(index);
}

system_provider
::system_provider()
{
}

system_provider
::~system_provider()
{
}

config::value_t
system_provider
::operator () (config::value_t const& index) const
{
  config::value_t value;

  if (index == "processors")
  {
    value = boost::lexical_cast<config::value_t>(boost::thread::hardware_concurrency());
  }
  else if (index == "homedir")
  {
    envvar_value_t const home = get_envvar(
#if defined(_WIN32) || defined(_WIN64)
      "UserProfile"
#else
      "HOME"
#endif
    );

    if (home)
    {
      value = config::value_t(*home);
    }
  }
  else if (index == "curdir")
  {
    boost::system::error_code ec;
    path_t const curdir = boost::filesystem::current_path(ec);

    /// \todo Check ec.

    value = curdir.string<config::value_t>();
  }
  else if (index == "pid")
  {
#if defined(_WIN32) || defined(_WIN64)
    int const pid = _getpid();
#else
    pid_t const pid = getpid();
#endif

    value = boost::lexical_cast<config::value_t>(pid);
  }
  else
  {
    throw unrecognized_system_index_exception(index);
  }

  return value;
}

environment_provider
::environment_provider()
{
}

environment_provider
::~environment_provider()
{
}

config::value_t
environment_provider
::operator () (config::value_t const& index) const
{
  envvar_name_t const envvar_name = index.c_str();
  envvar_value_t const envvar_value = get_envvar(envvar_name);

  config::value_t value;

  if (envvar_value)
  {
    value = config::value_t(*envvar_value);
  }

  return value;
}

}
