// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file pipe_bakery_exception.cxx
 *
 * \brief Implementations of exceptions used when baking a pipeline.
 */

#include "pipe_bakery_exception.h"

#include <viame/algorithm_framework/util/source_location.h>

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
