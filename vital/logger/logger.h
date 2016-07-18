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

#ifndef KWIVER_CORE_LOGGER_H_
#define KWIVER_CORE_LOGGER_H_

#include "kwiver_logger.h"

/**
 * @file
 * This file defines the main user interface to the kwiver logger.
 */


// ----------------------------------------------------------------
/** @page Logger Logger Documentation

<P>The kwiver logger (class kwiver_logger) provides a interface to an
underlying log implementation. Log4cxx is the baseline implementation,
which is why this interface looks the way it does. Alternate loggers
can be instantiated as needed for specific applications, products, or
projects. These alternate logger implementations are supplied by a
factory class and can provide any functionality consistent with the
kwiver_logger interface. The semantics of the underlying logger
largely pass through to this implementation.</P>

<P>All calls to log a message require a logger object which is
obtained through the kwiver::vital::get_logger( <name> ) call. The
supplied name is used to retrieve a logger object with that
name. Calls that use the same name will get the same logger
object. Depending on the underlying logger being used, more than one
name may map to the same logger object.</P>

<P>If Log4cxx is not available or not enabled, a minimal logger is
supplied which routes all log messages to standard error output
(std::err). This is a case where multiple names map to the same logger
object.</P>

<P>The easiest way to applying the logger is to use the macros in the
logger/logger.h file. It is most efficient to locally cache the logger
pointer in a class member variable.</p>

@sa kwiver_logger


<h2>Internal Operation</h2>

<p>During construction, the logger_manager instantiates a factory
class from a loadable module, named "vital_logger_plugin.so" (or
"vital_logger_plugin.dll" for windows). This plugin is expected to be
somewhere in the standard library loading path. If the logger plugin
has a different name or in a specific location, the environment
variable \b VITAL_LOGGER_FACTORY can be used to specify the name and
location (full path and file name). If a valid logger plugin can not
be found the default minimal logger is used.</P>

<p>Using a default plugin allows a logger factory to be supplied by
the installed set of libraries. This helps in cases where it is not
practical to set an environment variable. A log4cxx logger factory is
built by default and can be set up as the default plugin by renaming
it or using a symbolic link.</p>

<P>The ability to support alternate underlying logger implementations
is designed to make this logging component easy(er) to transport to
projects that have a specific (not log4cxx) logging implementation
requirement. Alternate logger factories are dynamically loaded at run
time.</p>

<P>An alternate logger back end is created by implementing a concrete
version of the logger interface derived from
kwiver::vital::logger_ns::kwiver_logger_factory class and a logger
factory class derived from
kwiver::vital::logger_ns::kwiver_logger_factory. Finally a bootstrap
function is needed by the plugin loader to get an instance of the
logger factory. Refer to logger/log4cxx_factory.cxx file for guidance.
</P>


<h2>Configuration</h2>

<h3>Log4cxx</H3>

The Log4cxx implementation uses a specific configuration file format that is described at:
<P> @link https://logging.apache.org/log4cxx/usage.html </p>

The configuration process is as follows:

- Look for the \b LOG4CXX_CONFIGURATION environment variable. If set,
  use this as the configuration file.

- If no environment variable set, look for the first of "log4cxx.xml",
"log4cxx.properties", "log4j.xml" and "log4j.properties" in the
current working directory and use that as the configuration file.

If still no configuration file can be found, then a default
configuration is used, which generally does not do what you really
want.

<h3>Other logger back ends</H3>

<P>Other underlying loggers may have different configuration procedures.</P>

<h2>Example</h2>

\code
#include <vital/logger/logger.h>
#include <iostream>

kwiver::vital::logger_handle_t m_logger;

int main(int argc, char *argv[])
{

  m_logger = kwiver::vital::get_logger( "main.logger" );

  LOG_ERROR( m_logger, "first message" << " from here");

  LOG_FATAL( m_logger, "fatal message");
  LOG_ERROR( m_logger, "error message");
  LOG_WARN ( m_logger, "warning message");
  LOG_INFO ( m_logger, "info message");
  LOG_DEBUG( m_logger, "debug message");
  LOG_TRACE( m_logger, "trace message");

  return 0;
}
\endcode

 */


namespace kwiver {
namespace vital {

//@{
/**
 * @brief Get pointer to logger object.
 *
 * @param name Logger name
 *
 * @return Handle (pointer) to logger object.
 */
logger_handle_t VITAL_LOGGER_EXPORT get_logger( const char * const name );
logger_handle_t VITAL_LOGGER_EXPORT get_logger( std::string const& name );
//@}

/**
 * Logs a message with the ERROR level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_ERROR( logger, msg ) do {                        \
    if ( logger->is_error_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                 \
      logger->log_error( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the WARN level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_WARN( logger, msg ) do {                        \
    if ( logger->is_warn_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                \
      logger->log_warn( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the INFO level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_INFO( logger, msg ) do {                        \
    if ( logger->is_info_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                \
      logger->log_info( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the DEBUG level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_DEBUG( logger, msg ) do {                        \
    if ( logger->is_debug_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                 \
      logger->log_debug( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )


/**
 * Logs a message with the TRACE level.
 * @param logger the logger to be used
 * @param msg the message string to log.
 */
#define LOG_TRACE( logger, msg ) do {                        \
    if ( logger->is_trace_enabled() ) {                      \
      std::stringstream _oss_; _oss_ << msg;                 \
      logger->log_trace( _oss_.str(), KWIVER_LOGGER_SITE ); } \
} while ( 0 )

/**
 * Performs assert and logs message if condition is false.  If
 * condition is false, log a message at the FATAL level is
 * generated. This is similar to the library assert except that the
 * message goes to the logger.
 * @param logger the logger to be used
 * @param cond the condition which should be met to pass the assertion
 * @param msg the message string to log.
 */
#define LOG_ASSERT( logger, cond, msg ) do {                   \
    if ( ! ( cond ) ) {                                        \
      std::stringstream _oss_; _oss_  << "ASSERTION FAILED: (" \
                                      << # cond ")\n"  << msg; \
      logger->log_error( _oss_.str(), KWIVER_LOGGER_SITE );     \
} while ( 0 )


// Test for debugging level being enabled
#define IS_FATAL_ENABLED( logger ) ( logger->is_fatal_enabled() )
#define IS_ERROR_ENABLED( logger ) ( logger->is_error_enabled() )
#define IS_WARN_ENABLED( logger )  ( logger->is_warn_enabled() )
#define IS_INFO_ENABLED( logger )  ( logger->is_info_enabled() )
#define IS_DEBUG_ENABLED( logger ) ( logger->is_debug_enabled() )
#define IS_TRACE_ENABLED( logger ) ( logger->is_trace_enabled() )

} } // end namespace

#endif /* KWIVER_CORE_LOGGER_H_ */
