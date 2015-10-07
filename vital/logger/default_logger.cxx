/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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


#include <vital/logger/vital_logger_export.h>
#include "default_logger.h"
#include "kwiver_logger.h"

#include <memory>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string.h>
#include <locale>

namespace kwiver {
namespace vital {
namespace logger_ns {


static void
writeTime( std::ostream& os, time_t const& tc )
{
  using namespace std;

  // alternative to put_time iomanip
  locale loc;
  const time_put< char >& tp = use_facet < time_put < char > > ( loc );
  const char* pat = "%F %T";
  tp.put( os, os, ' ', localtime( &tc ), pat, pat + strlen( pat ) );
}


// ------------------------------------------------------------------
logger_factory_default
::logger_factory_default()
  : kwiver_logger_factory( "default_logger factory" )
{
}


// ==================================================================
/**
 * @brief Default kwiver logger implementation.
 *
 * This class implements a default minimal logger that is instantiated
 * if there is no other logger back end.
 */
class default_logger :
  public kwiver_logger
{
public:
  /// CTOR
  default_logger( logger_ns::logger_factory_default* p, std::string const& name )
    : kwiver_logger( p, name ),
    m_logLevel( kwiver_logger::LEVEL_TRACE )
  { }

  virtual ~default_logger() VITAL_DEFAULT_DTOR

  // Check to see if level is enabled
  virtual bool is_fatal_enabled() const { return m_logLevel <= kwiver_logger::LEVEL_FATAL;  }

  virtual bool is_error_enabled() const { return m_logLevel <= kwiver_logger::LEVEL_ERROR;  }

  virtual bool is_warn_enabled()  const { return m_logLevel <= kwiver_logger::LEVEL_WARN;  }

  virtual bool is_info_enabled()  const { return m_logLevel <= kwiver_logger::LEVEL_INFO;  }

  virtual bool is_debug_enabled() const { return m_logLevel <= kwiver_logger::LEVEL_DEBUG;  }

  virtual bool is_trace_enabled() const { return m_logLevel <= kwiver_logger::LEVEL_TRACE;  }

  virtual void set_level( log_level_t lev ) { m_logLevel = lev; }

  virtual log_level_t get_level() const { return m_logLevel; }

  virtual void log_fatal( std::string const& msg )
  {
    if ( is_fatal_enabled() ) { log_message( LEVEL_FATAL, msg ); }
  }


  virtual void log_fatal( std::string const&              msg,
                          logger_ns::location_info const& location )
  {
    if ( is_fatal_enabled() ) { log_message( LEVEL_FATAL, msg, location ); }
  }


  virtual void log_error( std::string const& msg )
  {
    if ( is_error_enabled() ) { log_message( LEVEL_ERROR, msg ); }
  }


  virtual void log_error( std::string const&              msg,
                          logger_ns::location_info const& location )
  {
    if ( is_error_enabled() ) { log_message( LEVEL_ERROR, msg, location ); }
  }


  virtual void log_warn( std::string const& msg )
  {
    if ( is_warn_enabled() ) { log_message( LEVEL_WARN, msg ); }
  }


  virtual void log_warn( std::string const&               msg,
                         logger_ns::location_info const&  location )
  {
    if ( is_warn_enabled() ) { log_message( LEVEL_WARN, msg, location ); }
  }


  virtual void log_info( std::string const& msg )
  {
    if ( is_info_enabled() ) { log_message( LEVEL_INFO, msg ); }
  }


  virtual void log_info( std::string const&               msg,
                         logger_ns::location_info const&  location )
  {
    if ( is_info_enabled() ) { log_message( LEVEL_INFO, msg, location ); }
  }


  virtual void log_debug( std::string const& msg )
  {
    if ( is_debug_enabled() ) { log_message( LEVEL_DEBUG, msg ); }
  }


  virtual void log_debug( std::string const&              msg,
                          logger_ns::location_info const& location )
  {
    if ( is_debug_enabled() ) { log_message( LEVEL_DEBUG, msg, location ); }
  }


  virtual void log_trace( std::string const& msg )
  {
    if ( is_trace_enabled() ) { log_message( LEVEL_TRACE, msg ); }
  }


  virtual void log_trace( std::string const&              msg,
                          logger_ns::location_info const& location )
  {
    if ( is_trace_enabled() ) { log_message( LEVEL_TRACE, msg, location ); }
  }


private:
  // ------------------------------------------------------------------
  virtual void log_message( log_level_t         level,
                            std::string const&  msg )
  {
    static std::mutex lock;

    using namespace std::chrono;

    // Format this message on the stream

    // Get the current time in milliseconds, creating a formatted
    // string for log message.
    high_resolution_clock::time_point p = high_resolution_clock::now();

    milliseconds ms = duration_cast<milliseconds>(p.time_since_epoch());

    seconds s = duration_cast<seconds>(ms);
    std::time_t t = s.count();
    std::size_t fractional_seconds = ms.count() % 1000;

    // Ensure that multi-line messages still get the time and level prefix
    std::string  const level_str = get_level_string( level );
    std::string msg_part;
    std::istringstream ss( msg );

    {
      std::lock_guard< std::mutex > guard( lock ); // serialize access to stream
      std::ostream* str = &get_stream();
      while ( getline( ss, msg_part ) )
      {
        writeTime( *str, t );
        *str
          // << std::put_time(std::localtime(&t), "%F %T")
           << '.' << fractional_seconds
           << ' ' << level_str << ' ' << msg_part << '\n';
      }
    }
  }


  // ------------------------------------------------------------------
  virtual void log_message( log_level_t                     level,
                            std::string const&              msg,
                            logger_ns::location_info const& location )
  {
    log_message( level, msg );
  }


  // ------------------------------------------------------------------
  std::ostream& get_stream()
  {
    return *s_output_stream;
  }


  // ##################################################################
  log_level_t                  m_logLevel;       // current logging level

  std::mutex                   m_formatter_mtx;

  static std::ostream*         s_output_stream;

}; // end class logger


// -- STATIC data --
// Set up default logging stream
std::ostream* default_logger::s_output_stream = &std::cerr;


// ==================================================================
logger_handle_t
logger_factory_default
::get_logger( std::string const& name )
{
  // look for logger in the map
  std::map< std::string, logger_handle_t >::iterator it = m_active_loggers.find( name );
  if (it != m_active_loggers.end() )
  {
    return it->second;
  }

  return std::make_shared< default_logger > ( this, name );
}


}
}
}     // end namespace
