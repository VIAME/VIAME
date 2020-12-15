// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_DEFAULT_LOGGER_H_
#define KWIVER_DEFAULT_LOGGER_H_

#include <vital/vital_config.h>
#include "kwiver_logger_factory.h"

#include <map>
#include <string>

namespace kwiver {
namespace vital {
namespace logger_ns {

// ----------------------------------------------------------------
/**
 * @brief Factory for default underlying logger.
 *
 * This class represents the factory for the default logging service.
 *
 * This is a minimal implementation of a kwiver logger. The logging
 * level is set to a default threshold depending on the build mode. If
 * built in debug mode, the threshold is set to display TRACE level
 * messages and above, which is all lob messages. If built in release
 * mode, the threshold is set to display WARN level messages and
 * above, which results in a much reduced logger output.
 *
 * The default log threshold can be overridden by an environment
 * variable "KWIVER_DEFAULT_LOG_LEVEL". If this is set to a
 * recognizable level, the default will be set to that level. If the
 * level is not recognized, the default level is unchanged and no
 * error message is generated.
 *
 * Recognized levels are: trace, debug, info, warn, error, fatal.
 */
class logger_factory_default
  : public kwiver_logger_factory
{
public:
  logger_factory_default();
  virtual ~logger_factory_default() = default;

  /**
   * @brief Get logger object for /c name.
   *
   * This method returns a handle to the named logger. Since this is
   * the minimal default logger, all loggers are effectively the same.
   *
   * @param name Name of the logger.
   *
   * @return Handle to desired logger.
   */
  virtual logger_handle_t get_logger( std::string const& name );

private:
  std::map< std::string, logger_handle_t > m_active_loggers;

}; // end class logger_factory

} } } // end namespace

#endif
