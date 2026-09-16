// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_LOGGER_LOCATION_INFO_H_
#define KWIVER_LOGGER_LOCATION_INFO_H_

#include <viame/algorithm_framework/logger/vital_logger_export.h>

#include <string>

namespace viame {

namespace logger_ns {

// ----------------------------------------------------------------------------
/// Where a log call was made.
///
/// Captured by the `KWIVER_LOGGER_SITE` macro below, which the `LOG_*`
/// macros pass to the logger. The three pointers are to string literals the
/// preprocessor produced, so this stays copyable and costs nothing to pass.
class VITAL_LOGGER_EXPORT location_info
{
public:
  /// A location that is not known.
  location_info();

  /// The location the macro captured.
  location_info( char const* filename, char const* method, int line );

  //@{
  /// What an unknown location reports.
  static const char* const NA;
  static const char* const NA_METHOD;
  //@}

  /// @brief The file name, without its directories.
  ///
  /// This is what appears in a log line. `__FILE__` is whatever path the
  /// compiler was given, which is the build machine's and of no use to a
  /// reader, so only the last component is kept.
  std::string get_file_name() const;

  /// @brief The whole function signature the compiler gave.
  std::string get_signature() const;

  /// @brief The line the log call is on, or -1 if it is not known.
  int get_line_number() const;

private:
  const char* const m_fileName;
  const char* const m_methodName;
  int m_lineNumber;
}; // end class location_info

} // namespace logger_ns

} // namespace viame

#if defined( _MSC_VER )
#if _MSC_VER >= 1300
#define __KWIVER_LOGGER_FUNC__ __FUNCSIG__
#endif
#else
#if defined( __GNUC__ )
#define __KWIVER_LOGGER_FUNC__ __PRETTY_FUNCTION__
#endif
#endif
#if !defined( __KWIVER_LOGGER_FUNC__ )
#define __KWIVER_LOGGER_FUNC__ ""
#endif

#define KWIVER_LOGGER_SITE                 \
::viame::logger_ns::location_info( \
  __FILE__,                                \
  __KWIVER_LOGGER_FUNC__,                  \
  __LINE__ )

#endif
