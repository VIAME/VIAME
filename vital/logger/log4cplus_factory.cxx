/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include "kwiver_logger_factory.h"
#include <kwiversys/SystemTools.hxx>
#include <vital/logger/vital_log4cplus_logger_export.h>
#include <vital/util/get_paths.h>

#include <log4cplus/configurator.h>
#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include <sstream>

namespace kwiver {
namespace vital {
namespace logger_ns {

// ----------------------------------------------------------------
/**
 * @brief loc4cplus logger implementation
 *
 * This class is an adapter that bridges from our standard logging API
 * to the log4cplus API.
 */
class log4cplus_logger
  : public kwiver_logger
{
public:

log4cplus_logger( kwiver_logger_factory* fact, std::string const& name )
  : kwiver_logger( fact, name )
  , m_logger( ::log4cplus::Logger::getInstance( name ) )
{ }


~log4cplus_logger()
{ }


virtual bool is_fatal_enabled() const { return m_logger.isEnabledFor(::log4cplus::FATAL_LOG_LEVEL); }
virtual bool is_error_enabled() const { return m_logger.isEnabledFor(::log4cplus::ERROR_LOG_LEVEL); }
virtual bool is_warn_enabled() const { return m_logger.isEnabledFor(::log4cplus::WARN_LOG_LEVEL); }
virtual bool is_info_enabled() const { return m_logger.isEnabledFor(::log4cplus::INFO_LOG_LEVEL); }
virtual bool is_debug_enabled() const { return m_logger.isEnabledFor(::log4cplus::DEBUG_LOG_LEVEL); }
virtual bool is_trace_enabled() const { return m_logger.isEnabledFor(::log4cplus::TRACE_LOG_LEVEL); }


// ----------------------------------------------------------------
/* get / set log level
 *
 *
 */
virtual void set_level( log_level_t level )
{
  switch (level)
  {
  case LEVEL_TRACE:
    m_logger.setLogLevel(::log4cplus::TRACE_LOG_LEVEL);
    break;

  case LEVEL_DEBUG:
    m_logger.setLogLevel(::log4cplus::DEBUG_LOG_LEVEL);
    break;

  case LEVEL_INFO:
    m_logger.setLogLevel(::log4cplus::INFO_LOG_LEVEL);
    break;

  case LEVEL_WARN:
    m_logger.setLogLevel(::log4cplus::WARN_LOG_LEVEL);
    break;

  case LEVEL_ERROR:
    m_logger.setLogLevel(::log4cplus::ERROR_LOG_LEVEL);
    break;

  case LEVEL_FATAL:
    m_logger.setLogLevel(::log4cplus::FATAL_LOG_LEVEL);
    break;

  default:
    // log or throw
    break;
  } // end switch
}


// ----------------------------------------------------------------
virtual log_level_t get_level() const
{
  ::log4cplus::LogLevel lvl = this->m_logger.getLogLevel();

  if (lvl == ::log4cplus::TRACE_LOG_LEVEL) return LEVEL_TRACE;
  if (lvl == ::log4cplus::DEBUG_LOG_LEVEL) return LEVEL_DEBUG;
  if (lvl == ::log4cplus::INFO_LOG_LEVEL)  return LEVEL_INFO;
  if (lvl == ::log4cplus::WARN_LOG_LEVEL)  return LEVEL_WARN;
  if (lvl == ::log4cplus::ERROR_LOG_LEVEL) return LEVEL_ERROR;
  if (lvl == ::log4cplus::FATAL_LOG_LEVEL) return LEVEL_FATAL;

  return LEVEL_NONE;
}


// ----------------------------------------------------------------
virtual void log_fatal (std::string const & msg)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::FATAL_LOG_LEVEL,
    msg,
    0, 0, 0);
  do_callback(LEVEL_FATAL, msg, location_info());
}


// ----------------------------------------------------------------
  virtual void log_fatal (std::string const & msg,
                          kwiver::vital::logger_ns::location_info const & location)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::FATAL_LOG_LEVEL,
    msg,
    location.get_file_name_ptr(), location.get_line_number(),
    location.get_method_name_ptr());
  do_callback(LEVEL_FATAL, msg, location);
}


// ----------------------------------------------------------------
virtual void log_error (std::string const & msg)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::ERROR_LOG_LEVEL,
    msg,
    0, 0, 0);
  do_callback(LEVEL_ERROR, msg, location_info());
}


// ----------------------------------------------------------------
virtual void log_error (std::string const & msg,
                        kwiver::vital::logger_ns::location_info const & location)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::ERROR_LOG_LEVEL,
    msg,
    location.get_file_name_ptr(), location.get_line_number(),
    location.get_method_name_ptr());
  do_callback(LEVEL_ERROR, msg, location);
}


// ----------------------------------------------------------------
virtual void log_warn (std::string const & msg)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::WARN_LOG_LEVEL,
    msg,
    0, 0, 0);
  do_callback(LEVEL_WARN, msg, location_info());
}


// ----------------------------------------------------------------
virtual void log_warn (std::string const & msg,
                       kwiver::vital::logger_ns::location_info const & location)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::WARN_LOG_LEVEL,
    msg,
    location.get_file_name_ptr(), location.get_line_number(),
    location.get_method_name_ptr());
  do_callback(LEVEL_WARN, msg, location);
}


// ----------------------------------------------------------------
virtual void log_info (std::string const & msg)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::INFO_LOG_LEVEL,
    msg,
    0, 0, 0);
  do_callback(LEVEL_INFO, msg, location_info());
}


// ----------------------------------------------------------------
virtual void log_info (std::string const & msg,
                       kwiver::vital::logger_ns::location_info const & location)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::INFO_LOG_LEVEL,
    msg,
    location.get_file_name_ptr(), location.get_line_number(),
    location.get_method_name_ptr());
  do_callback(LEVEL_INFO, msg, location);
}


// ----------------------------------------------------------------
virtual void log_debug (std::string const & msg)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::DEBUG_LOG_LEVEL,
    msg,
    0, 0, 0);
  do_callback(LEVEL_DEBUG, msg, location_info());
}


// ----------------------------------------------------------------
virtual void log_debug (std::string const & msg,
                       kwiver::vital::logger_ns::location_info const & location)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    ::log4cplus::DEBUG_LOG_LEVEL,
    msg,
    location.get_file_name_ptr(), location.get_line_number(),
    location.get_method_name_ptr());
  do_callback(LEVEL_DEBUG, msg, location);
}


// ----------------------------------------------------------------
virtual void log_trace (std::string const & msg)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    log4cplus::TRACE_LOG_LEVEL,
    msg,
    0, 0, 0);
  do_callback(LEVEL_TRACE, msg, location_info());
}


// ----------------------------------------------------------------
virtual void log_trace (std::string const & msg,
                       kwiver::vital::logger_ns::location_info const & location)
{
  ::log4cplus::detail::macro_forced_log(m_logger,
    log4cplus::TRACE_LOG_LEVEL,
    msg,
    location.get_file_name_ptr(), location.get_line_number(),
    location.get_method_name_ptr());
  do_callback(LEVEL_TRACE, msg, location);
}


// ----------------------------------------------------------------
virtual void log_message ( log_level_t level, std::string const& msg)
{
  switch (level)
  {
  case LEVEL_TRACE:
    log_trace(msg);
    break;
  case LEVEL_DEBUG:
    log_debug(msg);
    break;
  case LEVEL_INFO:
    log_info(msg);
    break;
  case LEVEL_WARN:
    log_warn(msg);
    break;
  case LEVEL_ERROR:
    log_error(msg);
    break;
  case LEVEL_FATAL:
    log_fatal(msg);
    break;
  default:
    break;
  } // end switch
}


// ----------------------------------------------------------------
virtual void log_message ( log_level_t level, std::string const& msg,
                           kwiver::vital::logger_ns::location_info const & location)
{
  switch (level)
  {
  case LEVEL_TRACE:
    log_trace(msg, location);
    break;

  case LEVEL_DEBUG:
    log_debug(msg, location);
    break;

  case LEVEL_INFO:
    log_info(msg, location);
    break;

  case LEVEL_WARN:
    log_warn(msg, location);
    break;

  case LEVEL_ERROR:
    log_error(msg, location);
    break;

  case LEVEL_FATAL:
    log_fatal(msg, location);
    break;

  default:
    break;
  } // end switch
}

  protected:
// The native logger object
::log4cplus::Logger m_logger;

};


// ==================================================================
/** Factory for underlying log4cxx logger.
 *
 * This class represents the factory for the log4cxx logging service.
 * A logger object is created or reused for the specified name.
 */
class log4cplus_factory
  : public kwiver_logger_factory
{
public:
  log4cplus_factory()
    : kwiver_logger_factory( "log4cplus factory" )
  {
    using ST = kwiversys::SystemTools;
    std::string config_file;

    // Try the environemnt variable if no config file yet
    if ( ! ST::GetEnv( "LOG4CPLUS_CONFIGURATION", config_file ) )
    {
      auto const exe_path = kwiver::vital::get_executable_path();
      const std::string prop_file = "log4cplus.properties";
      if (ST::FileExists( "./" + prop_file) )
      {
        config_file = "./" + prop_file;
      }
      else if (ST::FileExists(exe_path + "/" + prop_file))
      {
        config_file = exe_path + "/" + prop_file;
      }
      else if (ST::FileExists(exe_path + "/../lib/kwiver/" + prop_file))
      {
        config_file = exe_path + "/../lib/kwiver/" + prop_file;
      }
    }

    if ( ! config_file.empty())
    {
      ::log4cplus::PropertyConfigurator::doConfigure(config_file);
    }
    else
    {
      ::log4cplus::BasicConfigurator::doConfigure();
    }
  }


  virtual ~log4cplus_factory() = default;

  virtual logger_handle_t get_logger( std::string const& name )
  {
    return logger_handle_t( new log4cplus_logger ( this, name ) );
  }

}; // end class log4cplus_factory

} } } // end namespace


// ==================================================================
/*
 * Shared object bootstrap function
 */
extern "C" VITAL_LOG4CPLUS_LOGGER_EXPORT void* kwiver_logger_factory();

void* kwiver_logger_factory()
{
  kwiver::vital::logger_ns::log4cplus_factory* ptr =  new kwiver::vital::logger_ns::log4cplus_factory();
  return ptr;
}
