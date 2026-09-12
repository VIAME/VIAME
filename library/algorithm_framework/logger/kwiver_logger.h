// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_KWIVER_LOGGER_H_
#define KWIVER_KWIVER_LOGGER_H_

#include "location_info.h"
#include <viame/algorithm_framework/logger/vital_logger_export.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>

#include <viame/core_types/noncopyable.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// @brief A named logger.
///
/// Get one with `get_logger( name )` and log through the `LOG_*` macros in
/// `logger.h`; those are the interface, and this class is what they call.
///
/// The name selects the logger and its level. It does not appear in the
/// output -- a callback is how a program gets at it, and at the source
/// location, which the formatted line also drops.
///
/// This used to be an abstract base with a factory behind it, loaded from a
/// shared library named by `VITAL_LOGGER_FACTORY`, so that a project could
/// substitute log4cxx or log4cplus. P8-T04 removed all of that: the two
/// implementations that existed were behind build options that have never
/// been on, VIAME ships one logger, and the indirection was costing a
/// `dlopen` at static-initialisation time on every process start.
class VITAL_LOGGER_EXPORT kwiver_logger
  : public std::enable_shared_from_this< kwiver_logger >,
    private kwiver::vital::noncopyable
{
public:
  enum log_level_t
  {
    LEVEL_NONE = 1,
    LEVEL_TRACE,
    LEVEL_DEBUG,
    LEVEL_INFO,
    LEVEL_WARN,
    LEVEL_ERROR,
    LEVEL_FATAL,
  };

/// @brief Make a logger.
///
/// Call `get_logger()` rather than this: loggers are shared by name, and one
/// made here is not in that table.
  explicit kwiver_logger( std::string const& name );

  ~kwiver_logger();

// Is this level worth formatting a message for? The `LOG_*` macros ask
// first, which is what makes a disabled `LOG_TRACE` cost nothing but the
// comparison.
  bool is_fatal_enabled() const;
  bool is_error_enabled() const;
  bool is_warn_enabled()  const;
  bool is_info_enabled()  const;
  bool is_debug_enabled() const;
  bool is_trace_enabled() const;

/// @brief Set the level this logger writes at, overriding the environment.
  void set_level( log_level_t lev );

  log_level_t get_level() const;

/// Type alias for the callback function signature
  using callback_t =
    std::function< void ( log_level_t, std::string const& name,
                          std::string const& msg,
                          logger_ns::location_info const& loc ) >;

/// Set a callback to be called on logging events for this logger instance
  void set_local_callback( callback_t cb );

/// Set a callback to be called on logging events for all logger instances
  static void set_global_callback( callback_t cb );

/// @brief Get logger name.
  std::string get_name() const;

//@{
/// @brief Log a message at this level.
///
/// The message is written if the level is enabled, and dropped otherwise.
/// The overload taking a location is what the `LOG_*` macros call.
  void log_fatal( std::string const& msg );
  void log_fatal(
    std::string const& msg, logger_ns::location_info const& location );

  void log_error( std::string const& msg );
  void log_error(
    std::string const& msg, logger_ns::location_info const& location );

  void log_warn( std::string const& msg );
  void log_warn(
    std::string const& msg, logger_ns::location_info const& location );

  void log_info( std::string const& msg );
  void log_info(
    std::string const& msg, logger_ns::location_info const& location );

  void log_debug( std::string const& msg );
  void log_debug(
    std::string const& msg, logger_ns::location_info const& location );

  void log_trace( std::string const& msg );
  void log_trace(
    std::string const& msg, logger_ns::location_info const& location );

  void log_message( log_level_t level, std::string const& msg );
  void log_message(
    log_level_t level, std::string const& msg,
    logger_ns::location_info const& location );
//@}

/// @brief Convert level code to string.
///
/// @param lev level value to convert
  static char const* get_level_string( kwiver_logger::log_level_t lev );

private:
  class impl;

  const std::unique_ptr< impl > m_impl;
}; // end class logger

/// @brief Handle for kwiver logger objects.
typedef std::shared_ptr< kwiver_logger > logger_handle_t;

} // namespace vital

}   // end namespace

#endif
