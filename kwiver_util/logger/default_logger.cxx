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


#include <logger/kwiver_logger_export.h>
#include "default_logger.h"
#include "kwiver_logger.h"

#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/date_time.hpp>
#include <boost/ptr_container/ptr_map.hpp>

using namespace boost::posix_time;

#if defined LOADABLE_MODULE
// ------------------------------------------------------------------
/*
 * Shared object bootstrap function
 */
extern "C"
{
  void* KWIVER_LOGGER_EXPORT kwiver_logger_factory()
  {
    kwiver::logger_ns::logger_factory_default* ptr =  new kwiver::logger_ns::logger_factory_default;
    return ptr;
  }

}
#endif


namespace kwiver {
namespace logger_ns {

  // ==================================================================
  // Use 1 mutex per stream.  This needs to be static to allow for multiple
  // loggers to use the same stream and still have it locked appropriately
  boost::mutex& get_stream_mtx( const std::ostream& s )
  {
    static boost::shared_mutex stream_mtx_map_mtx;
    static boost::ptr_map< const std::ostream*, boost::mutex > stream_mtx_map;

    boost::shared_lock< boost::shared_mutex > stream_mtx_map_lock( stream_mtx_map_mtx );

    // create a new mutex if not already there
    if ( 0 == stream_mtx_map.count( &s ) )
    {
      boost::upgrade_lock< boost::shared_mutex > lock( stream_mtx_map_mtx );
      const std::ostream* tsp = &s;
      stream_mtx_map.insert( tsp, new boost::mutex() );
    }

    return stream_mtx_map[&s];
  }




// ------------------------------------------------------------------
logger_factory_default
::logger_factory_default()
  : kwiver_logger_factory( "default_logger factory" )
{
}


logger_factory_default
::~logger_factory_default()
{ }


// ----------------------------------------------------------------
/**
 * @brief kwiver logger interface
 *
 */
class default_logger
  : public kwiver_logger
{
public:

  default_logger( logger_ns::logger_factory_default* p, const char * const name )
    : kwiver_logger( p, name ),
      m_logLevel(kwiver_logger::LEVEL_TRACE)
  { }

  virtual ~default_logger() { }

  // Check to see if level is enabled
  virtual bool is_fatal_enabled() const { return (m_logLevel <= kwiver_logger::LEVEL_FATAL); }
  virtual bool is_error_enabled() const { return (m_logLevel <= kwiver_logger::LEVEL_ERROR); }
  virtual bool is_warn_enabled()  const { return (m_logLevel <= kwiver_logger::LEVEL_WARN); }
  virtual bool is_info_enabled()  const { return (m_logLevel <= kwiver_logger::LEVEL_INFO); }
  virtual bool is_debug_enabled() const { return (m_logLevel <= kwiver_logger::LEVEL_DEBUG); }
  virtual bool is_trace_enabled() const { return (m_logLevel <= kwiver_logger::LEVEL_TRACE); }

  virtual void set_level( log_level_t lev) { m_logLevel = lev; }
  virtual log_level_t get_level() const { return m_logLevel; }

  virtual void log_fatal (std::string const & msg)
  {
    if (is_fatal_enabled()) { log_message (LEVEL_FATAL, msg); }
  }

  virtual void log_fatal (std::string const & msg,
                          kwiver::logger_ns::location_info const & location)
  {
    if (is_fatal_enabled()) { log_message (LEVEL_FATAL, msg, location); }
  }

  virtual void log_error (std::string const & msg)
  {
    if (is_error_enabled()) { log_message (LEVEL_ERROR, msg); }
  }

  virtual void log_error (std::string const & msg,
                          kwiver::logger_ns::location_info const & location)
  {
    if (is_error_enabled()) { log_message (LEVEL_ERROR, msg, location); }
  }

  virtual void log_warn (std::string const & msg)
  {
    if (is_warn_enabled()) { log_message (LEVEL_WARN, msg); }
  }
  virtual void log_warn (std::string const & msg,
                         kwiver::logger_ns::location_info const & location)
  {
    if (is_warn_enabled()) { log_message (LEVEL_WARN, msg, location); }
  }

  virtual void log_info (std::string const & msg)
  {
    if (is_info_enabled()) { log_message (LEVEL_INFO, msg); }
  }

  virtual void log_info (std::string const & msg,
                         kwiver::logger_ns::location_info const & location)
  {
    if (is_info_enabled()) { log_message (LEVEL_INFO, msg, location); }
  }

  virtual void log_debug (std::string const & msg)
  {
    if (is_debug_enabled()) { log_message (LEVEL_DEBUG, msg); }
  }

  virtual void log_debug (std::string const & msg,
                          kwiver::logger_ns::location_info const & location)
  {
    if (is_debug_enabled()) { log_message (LEVEL_DEBUG, msg, location); }
  }

  virtual void log_trace (std::string const & msg)
  {
    if (is_trace_enabled()) { log_message (LEVEL_TRACE, msg); }
  }

  virtual void log_trace (std::string const & msg,
                          kwiver::logger_ns::location_info const & location)
  {
    if (is_trace_enabled()) { log_message (LEVEL_TRACE, msg, location); }
  }

private:

  // ------------------------------------------------------------------
  virtual void log_message (log_level_t level,
                            std::string const& msg)
  {
    // Format this message on the stream

    // Get the current time in milliseconds, creating a formatted
    // string for log message.
    ptime now = microsec_clock::local_time();

    // Ensure that multi-line messages still get the time and level prefix
    std::string level_str = get_level_string(level);
    std::string msg_part;
    std::istringstream ss(msg);

    std::ostream *s = &get_stream();
    {
      boost::lock_guard<boost::mutex> stream_lock(get_stream_mtx(*s));
      while(getline(ss, msg_part))
      {
        *s << now << ' ' << level_str << ' ' << msg_part << '\n';
      }
    }
  }

  // ------------------------------------------------------------------
  virtual void log_message (log_level_t level,
                            std::string const& msg,
                            kwiver::logger_ns::location_info const & location)
  {
    log_message( level, msg );
  }

  // ------------------------------------------------------------------
  std::ostream& get_stream()
  {
    // Make sure that any given stream only get's "imbued" once
    static std::map<std::ostream*, bool> is_imbued;
    static boost::mutex ctor_mtx;
    boost::lock_guard<boost::mutex> ctor_lock( ctor_mtx );

    if (!is_imbued[s_output_stream])
    {
      // Configure timestamp formatting
      time_facet* f = new time_facet("%Y-%m-%d %H:%M:%s");
      std::locale loc(std::locale(), f);
      {
        boost::lock_guard<boost::mutex> stream_lock( get_stream_mtx( *s_output_stream ) );
        s_output_stream->imbue(loc);
      }
      is_imbued[s_output_stream] = true;
    }

    return *s_output_stream;
  }

  // ##################################################################
  log_level_t                  m_logLevel;       // current logging level

  boost::mutex                 m_formatter_mtx;

  static std::ostream*         s_output_stream;

}; // end class logger


// -- STATIC data --
// Set up default logging stream
std::ostream* default_logger::s_output_stream = &std::cerr;


// ==================================================================
logger_handle_t
logger_factory_default
::get_logger( const char * const name )
{
  return boost::make_shared< default_logger >( this, name );
}

} // end namespace
} // end namespace
