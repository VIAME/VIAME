/*ckwg +29
 * Copyright 2017, 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "logger_process_instrumentation.h"

#include <sprokit/pipeline/process.h>
#include <vital/config/config_difference.h>
#include <vital/util/enum_converter.h>
#include <vital/util/string.h>

namespace sprokit {

using kvll = kwiver::vital::kwiver_logger::log_level_t;

  ENUM_CONVERTER( level_converter, kvll,
    { "trace", kvll::LEVEL_TRACE },
    { "debug", kvll::LEVEL_DEBUG },
    { "info",  kvll::LEVEL_INFO },
    { "warn",  kvll::LEVEL_WARN },
    { "error", kvll::LEVEL_ERROR }
  );

// ------------------------------------------------------------------
logger_process_instrumentation::
logger_process_instrumentation()
  : m_logger( kwiver::vital::get_logger( "sprokit.process_instrumentation" ) )
  , m_log_level( kvll::LEVEL_INFO )
{ }


// ------------------------------------------------------------------
void
logger_process_instrumentation::
  start_init_processing( std::string const& data )
{
  log_message( process()->name() + ": start_init_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_init_processing()
{
  log_message( process()->name() + ": stop_init_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
  start_finalize_processing( std::string const& data )
{
  log_message( process()->name() + ": start_finalize_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_finalize_processing()
{
  log_message( process()->name() + ": stop_finalize_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_reset_processing( std::string const& data )
{
  log_message( process()->name() + ": stop_init_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_reset_processing()
{
  log_message( process()->name() + ": stop_reset_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_flush_processing( std::string const& data )
{
  log_message( process()->name() + ": start_reset_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_flush_processing()
{
  log_message( process()->name() + ": stop_flush_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_step_processing( std::string const& data )
{
  log_message( process()->name() + ": start_step_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_step_processing()
{
  log_message( process()->name() + ": stop_step_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_configure_processing( std::string const& data )
{
  log_message( process()->name() + ": start_configure_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_configure_processing()
{
  log_message( process()->name() + ": stop_configure_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
start_reconfigure_processing( std::string const& data )
{
  log_message( process()->name() + ": start_reconfigure_processing" );
}


// ------------------------------------------------------------------
void
logger_process_instrumentation::
stop_reconfigure_processing()
{
  log_message( process()->name() + ": stop_reconfigure_processing" );
}

// ----------------------------------------------------------------------------
void
logger_process_instrumentation::
configure( kwiver::vital::config_block_sptr const conf )
{
  kwiver::vital::config_difference cd( this->get_configuration(), conf );
  const auto key_list = cd.extra_keys();
  if ( ! key_list.empty() )
  {
    LOG_WARN( m_logger, "Additional parameters found in config block that are not required or desired: "
              << kwiver::vital::join( key_list, ", " ) );
  }

  m_log_level = conf->get_enum_value<level_converter>( "level" );
}


// ----------------------------------------------------------------------------
kwiver::vital::config_block_sptr
logger_process_instrumentation::
get_configuration() const
{
  auto conf = kwiver::vital::config_block::empty_config();

  conf->set_value( "level", level_converter().to_string( m_log_level ),
                   "Logger level to use when generating log messages. "
                   "Allowable values are: " + level_converter().element_name_string()
    );


  return conf;
}


// ----------------------------------------------------------------------------
void logger_process_instrumentation
::log_message( const std::string& data )
{
  switch (m_log_level)
  {
  case kvll::LEVEL_TRACE:
    LOG_TRACE( m_logger, data );
    break;

  case kvll::LEVEL_DEBUG:
    LOG_DEBUG( m_logger, data );
    break;

  default:
  case kvll::LEVEL_INFO:
    LOG_INFO( m_logger, data );
    break;

  case kvll::LEVEL_WARN:
    LOG_WARN( m_logger, data );
    break;

  case kvll::LEVEL_ERROR:
    LOG_ERROR( m_logger, data );
    break;

  } // end switch
}

} // end namespace
