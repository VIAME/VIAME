/*ckwg +29
 * Copyright 2011-2013 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "providers.h"

#include "path.h"
#include "pipe_bakery_exception.h"

#include <vital/config/config_block.h>
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
::config_provider(kwiver::vital::config_block_sptr const conf)
  : m_config(conf)
{
}

config_provider
::~config_provider()
{
}

kwiver::vital::config_block_value_t
config_provider
::operator () (kwiver::vital::config_block_value_t const& index) const
{
  return m_config->get_value<kwiver::vital::config_block_value_t>(index);
}

system_provider
::system_provider()
{
}

system_provider
::~system_provider()
{
}

kwiver::vital::config_block_value_t
system_provider
::operator () (kwiver::vital::config_block_value_t const& index) const
{
  kwiver::vital::config_block_value_t value;

  if (index == "processors")
  {
    value = boost::lexical_cast<kwiver::vital::config_block_value_t>(boost::thread::hardware_concurrency());
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
      value = kwiver::vital::config_block_value_t(*home);
    }
  }
  else if (index == "curdir")
  {
    boost::system::error_code ec;
    path_t const curdir = boost::filesystem::current_path(ec);

    /// \todo Check ec.

    value = curdir.string<kwiver::vital::config_block_value_t>();
  }
  else if (index == "pid")
  {
#if defined(_WIN32) || defined(_WIN64)
    int const pid = _getpid();
#else
    pid_t const pid = getpid();
#endif

    value = boost::lexical_cast<kwiver::vital::config_block_value_t>(pid);
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

kwiver::vital::config_block_value_t
environment_provider
::operator () (kwiver::vital::config_block_value_t const& index) const
{
  envvar_name_t const envvar_name = index.c_str();
  envvar_value_t const envvar_value = get_envvar(envvar_name);

  kwiver::vital::config_block_value_t value;

  if (envvar_value)
  {
    value = kwiver::vital::config_block_value_t(*envvar_value);
  }

  return value;
}

}
