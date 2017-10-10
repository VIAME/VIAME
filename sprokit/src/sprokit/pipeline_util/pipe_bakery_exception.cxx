/*ckwg +29
 * Copyright 2011-2017 by Kitware, Inc.
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

/**
 * \file pipe_bakery_exception.cxx
 *
 * \brief Implementations of exceptions used when baking a pipeline.
 */

#include "pipe_bakery_exception.h"

#include <vital/util/source_location.h>

#include <sstream>

namespace sprokit {

// ------------------------------------------------------------------
pipe_bakery_exception
::pipe_bakery_exception() noexcept
  : pipeline_exception()
{
}


pipe_bakery_exception
::~pipe_bakery_exception() noexcept
{
}


// ------------------------------------------------------------------
missing_cluster_block_exception
::missing_cluster_block_exception() noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "A cluster block was not given when baking a cluster";
  m_what = sstr.str();
}


missing_cluster_block_exception
::~missing_cluster_block_exception() noexcept
{
}


// ------------------------------------------------------------------
multiple_cluster_blocks_exception
::multiple_cluster_blocks_exception() noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "Multiple cluster blocks were given when baking a cluster";
  m_what = sstr.str();
}


multiple_cluster_blocks_exception
::~multiple_cluster_blocks_exception() noexcept
{
}


// ------------------------------------------------------------------
cluster_without_processes_exception
::cluster_without_processes_exception() noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "A cluster cannot be baked without any processes";
  m_what = sstr.str();
}


cluster_without_processes_exception
::~cluster_without_processes_exception() noexcept
{
}


// ------------------------------------------------------------------
cluster_without_ports_exception
::cluster_without_ports_exception() noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << "A cluster cannot be baked without any ports";
  m_what = sstr.str();
}


cluster_without_ports_exception
::~cluster_without_ports_exception() noexcept
{
}


// ------------------------------------------------------------------
duplicate_cluster_port_exception
::duplicate_cluster_port_exception( process::port_t const& port,
                                    char const* const      side ) noexcept
  : pipe_bakery_exception(),
  m_port( port )
{
  std::stringstream sstr;

  sstr << "The " << side << " port "
                            "\'" << port << "\' was declared "
                                            "twice in a cluster";

  m_what = sstr.str();
}


duplicate_cluster_port_exception
::~duplicate_cluster_port_exception() noexcept
{
}


// ------------------------------------------------------------------
duplicate_cluster_input_port_exception
::duplicate_cluster_input_port_exception( process::port_t const& port ) noexcept
  : duplicate_cluster_port_exception( port, "input" )
{
}


duplicate_cluster_input_port_exception
::~duplicate_cluster_input_port_exception() noexcept
{
}


// ------------------------------------------------------------------
duplicate_cluster_output_port_exception
::duplicate_cluster_output_port_exception( process::port_t const& port ) noexcept
  : duplicate_cluster_port_exception( port, "output" )
{
}


duplicate_cluster_output_port_exception
::~duplicate_cluster_output_port_exception() noexcept
{
}


// ------------------------------------------------------------------
unrecognized_config_flag_exception
::unrecognized_config_flag_exception( kwiver::vital::config_block_key_t const& key, config_flag_t const& flag ) noexcept
  : pipe_bakery_exception(),
  m_key( key ),
  m_flag( flag )
{
  std::stringstream sstr;

  sstr << "The \'" << m_key << "\' key "
                               "has the \'" << m_flag << "\' on it "
                                                         "which is unrecognized";

  m_what = sstr.str();
}


unrecognized_config_flag_exception
::~unrecognized_config_flag_exception() noexcept
{
}


// ------------------------------------------------------------------
config_flag_mismatch_exception
::config_flag_mismatch_exception( kwiver::vital::config_block_key_t const& key,
                                  std::string const&                       reason ) noexcept
  : pipe_bakery_exception()
  , m_key( key )
  , m_reason( reason )
{
  std::stringstream sstr;

  sstr  << "The \'" << m_key << "\' key "
                               "has unsupported flags: "
        << m_reason;

  m_what = sstr.str();
}


config_flag_mismatch_exception
::~config_flag_mismatch_exception() noexcept
{
}


// ------------------------------------------------------------------
relativepath_exception
::relativepath_exception( const std::string&                    msg,
                          const kwiver::vital::source_location& loc ) noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << msg << " at " << loc;
  m_what = sstr.str();
}


relativepath_exception::
  ~relativepath_exception() noexcept
{ }


// ------------------------------------------------------------------
provider_error_exception::
provider_error_exception( const std::string&                    msg,
                          const kwiver::vital::source_location& loc ) noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << msg << " at " << loc;
  m_what = sstr.str();
}


  provider_error_exception::
  provider_error_exception( const std::string& msg ) noexcept
  : pipe_bakery_exception()
{
  std::stringstream sstr;

  sstr << msg;
  m_what = sstr.str();
}


provider_error_exception::
  ~provider_error_exception() noexcept
{ }

} // end namespace
