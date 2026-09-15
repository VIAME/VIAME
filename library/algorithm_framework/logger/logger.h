// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_CORE_LOGGER_H_
#define KWIVER_CORE_LOGGER_H_

#include "kwiver_logger.h"

/// @file
/// This file defines the main user interface to the kwiver logger.

// ----------------------------------------------------------------------------
/// @page Logger Logger Documentation
///
/// <P>All calls to log a message require a logger object, obtained with
/// kwiver::vital::get_logger( <name> ). Every call with the same name gets
/// the same logger, so it is cheapest to fetch the handle once and keep it in
/// a member. The macros below -- LOG_ERROR, LOG_WARN and the rest -- test the
/// level before they build the message.</P>
///
/// <P>There is one implementation, and nothing is loaded at run time to
/// provide it. Each line goes to standard error as
/// <tt>YYYY-MM-DD HH:MM:SS.mmm LEVEL file(line): message</tt>, local time. A
/// logger may also have a callback, and all loggers a global one, which see
/// each message instead of the default output.</P>
///
/// @sa kwiver_logger
///
/// <h2>Configuration</h2>
///
/// Two environment variables, read when a logger is first created:
///
/// - \b VIAME_LOG_LEVEL -- trace, debug, info, warn or error, in any case.
///   \b KWIVER_DEFAULT_LOG_LEVEL, the older name, is read when it is unset.
///   A value that names no level is ignored. Without either, a release build
///   logs warnings and errors and a debug build logs everything.
///
/// - \b VIAME_LOG_FILE -- a file to append every line to as well as standard
///   error. If it cannot be opened, the process says so once and logs to
///   standard error only.
///
/// A level set on a logger with set_level() overrides the environment for
/// that logger.
///
/// <h2>Example</h2>
///
/// \code
/// #include <viame/algorithm_framework/logger/logger.h>
/// #include <iostream>
///
/// kwiver::vital::logger_handle_t m_logger;
///
/// int main(int argc, char *argv[])
/// {
///
/// m_logger = kwiver::vital::get_logger( "main.logger" );
///
/// LOG_ERROR( m_logger, "first message" << " from here");
///
/// LOG_FATAL( m_logger, "fatal message");
/// LOG_ERROR( m_logger, "error message");
/// LOG_WARN ( m_logger, "warning message");
/// LOG_INFO ( m_logger, "info message");
/// LOG_DEBUG( m_logger, "debug message");
/// LOG_TRACE( m_logger, "trace message");
///
/// return 0;
/// }
/// \endcode
///

namespace kwiver {

namespace vital {

//@{
/// @brief Get pointer to logger object.
///
/// @param name Logger name
///
/// @return Handle (pointer) to logger object.
logger_handle_t VITAL_LOGGER_EXPORT get_logger( const char* const name );
logger_handle_t VITAL_LOGGER_EXPORT get_logger( std::string const& name );
//@}

/// Logs a message with the ERROR level.
/// @param logger the logger to be used
/// @param msg the message string to log.
#define LOG_ERROR( logger, msg )                          \
do                                                        \
{                                                         \
  if( logger->is_error_enabled() )                        \
  {                                                       \
    std::stringstream _oss_; _oss_ << msg;                \
    logger->log_error( _oss_.str(), KWIVER_LOGGER_SITE ); \
  }                                                       \
} while( 0 )

/// Logs a message with the WARN level.
/// @param logger the logger to be used
/// @param msg the message string to log.
#define LOG_WARN( logger, msg )                          \
do                                                       \
{                                                        \
  if( logger->is_warn_enabled() )                        \
  {                                                      \
    std::stringstream _oss_; _oss_ << msg;               \
    logger->log_warn( _oss_.str(), KWIVER_LOGGER_SITE ); \
  }                                                      \
} while( 0 )

/// Logs a message with the INFO level.
/// @param logger the logger to be used
/// @param msg the message string to log.
#define LOG_INFO( logger, msg )                          \
do                                                       \
{                                                        \
  if( logger->is_info_enabled() )                        \
  {                                                      \
    std::stringstream _oss_; _oss_ << msg;               \
    logger->log_info( _oss_.str(), KWIVER_LOGGER_SITE ); \
  }                                                      \
} while( 0 )

/// Logs a message with the DEBUG level.
/// @param logger the logger to be used
/// @param msg the message string to log.
#define LOG_DEBUG( logger, msg )                          \
do                                                        \
{                                                         \
  if( logger->is_debug_enabled() )                        \
  {                                                       \
    std::stringstream _oss_; _oss_ << msg;                \
    logger->log_debug( _oss_.str(), KWIVER_LOGGER_SITE ); \
  }                                                       \
} while( 0 )

/// Logs a message with the TRACE level.
/// @param logger the logger to be used
/// @param msg the message string to log.
#define LOG_TRACE( logger, msg )                          \
do                                                        \
{                                                         \
  if( logger->is_trace_enabled() )                        \
  {                                                       \
    std::stringstream _oss_; _oss_ << msg;                \
    logger->log_trace( _oss_.str(), KWIVER_LOGGER_SITE ); \
  }                                                       \
} while( 0 )

/// Performs assert and logs message if condition is false.  If
/// condition is false, log a message at the FATAL level is
/// generated. This is similar to the library assert except that the
/// message goes to the logger.
/// @param logger the logger to be used
/// @param cond the condition which should be met to pass the assertion
/// @param msg the message string to log.
#define LOG_ASSERT( logger, cond, msg )                    \
do                                                         \
{                                                          \
  if( !( cond ) )                                          \
  {                                                        \
    std::stringstream _oss_;                               \
    _oss_ << "ASSERTION FAILED: (" << #cond ")\n"  << msg; \
    logger->log_error( _oss_.str(), KWIVER_LOGGER_SITE );  \
  }                                                        \
} while( 0 )

// Test for debugging level being enabled
#define IS_FATAL_ENABLED( logger ) ( logger->is_fatal_enabled() )
#define IS_ERROR_ENABLED( logger ) ( logger->is_error_enabled() )
#define IS_WARN_ENABLED( logger ) ( logger->is_warn_enabled() )
#define IS_INFO_ENABLED( logger ) ( logger->is_info_enabled() )
#define IS_DEBUG_ENABLED( logger ) ( logger->is_debug_enabled() )
#define IS_TRACE_ENABLED( logger ) ( logger->is_trace_enabled() )

} // namespace vital

}   // end namespace

#endif
