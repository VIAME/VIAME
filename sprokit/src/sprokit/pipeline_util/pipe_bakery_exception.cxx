// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
