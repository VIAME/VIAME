/*ckwg +29
 * Copyright 2015-2017, 2019 by Kitware, Inc.
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
