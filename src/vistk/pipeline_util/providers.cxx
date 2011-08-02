/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "providers.h"

#include <vistk/pipeline/config.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <cstdlib>
#endif

/**
 * \file providers.cxx
 *
 * \brief Implementation of configuration providers.
 */

namespace vistk
{

namespace
{

typedef char const* envvar_name_t;
#if defined(_WIN32) || defined(_WIN64)
typedef char* envvar_value_t;
#else
typedef char const* envvar_value_t;
#endif

}

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
  return m_config->get_value<config::value_t>(index, config::value_t());
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
  /// \todo What keys do we want to provide here?

  return config::value_t();
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
  envvar_value_t envvar_value = NULL;
  envvar_name_t const envvar_name = index.c_str();

#if defined(_WIN32) || defined(_WIN64)
  DWORD sz = GetEnvironmentVariable(envvar_name, NULL, 0);

  if (sz)
  {
    envvar_value = new char[sz];

    sz = GetEnvironmentVariable(envvar_name, envvar_value, sz);
  }

  if (!sz)
  {
    /// \todo Log error that the environment reading failed.
  }
#else
  envvar_value = getenv(envvar_name);
#endif

  config::value_t value;

  if (envvar_value)
  {
    value = config::value_t(envvar_value);
  }

#if defined(_WIN32) || defined(_WIN64)
  delete [] envvar_value;
#endif

  return value;
}

}
