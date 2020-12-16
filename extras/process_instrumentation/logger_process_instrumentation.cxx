// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "logger_process_instrumentation.h"

#include <sprokit/pipeline/process.h>
#include <vital/vital_config.h>
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
  start_init_processing( VITAL_UNUSED std::string const& data )
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
  start_finalize_processing( VITAL_UNUSED std::string const& data )
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
start_reset_processing( VITAL_UNUSED std::string const& data )
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
start_flush_processing( VITAL_UNUSED std::string const& data )
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
start_step_processing( VITAL_UNUSED std::string const& data )
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
start_configure_processing( VITAL_UNUSED std::string const& data )
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
start_reconfigure_processing( VITAL_UNUSED std::string const& data )
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
  case kvll::LEVEL_NONE:
  case kvll::LEVEL_INFO:
    LOG_INFO( m_logger, data );
    break;

  case kvll::LEVEL_WARN:
    LOG_WARN( m_logger, data );
    break;

  case kvll::LEVEL_ERROR:
  case kvll::LEVEL_FATAL:
    LOG_ERROR( m_logger, data );
    break;

  } // end switch
}

} // end namespace
