/*ckwg +29
 * Copyright 2015, 2019 by Kitware, Inc.
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

#ifndef KWIVER_KWIVER_LOGGER_MANAGER_H_
#define KWIVER_KWIVER_LOGGER_MANAGER_H_

#include "kwiver_logger.h"
#include <vital/logger/vital_logger_export.h>

#include <string>
#include <memory>
#include <vital/noncopyable.h>

namespace kwiver {
namespace vital {
namespace logger_ns {

  class kwiver_logger_factory;

}

// ----------------------------------------------------------------
/** Logger manager.
 *
 * This class represents the main top level logic for the KWIVER
 * logger. Only one object of this type is required, so this is a
 * singleton created by the static instance() method.
 */
class VITAL_LOGGER_EXPORT kwiver_logger_manager
  :private kwiver::vital::noncopyable
{
public:
  ~kwiver_logger_manager();

  /** Get the single instance of this class. */
  static kwiver_logger_manager * instance();

  /**
   * @brief Get name of current logger factory.
   *
   * This method returns the name of the currently active logger
   * factory.
   *
   * @return Name of logger factory.
   */
  std::string const&  get_factory_name() const;

  /**
   * @brief Establish a new logger factory.
   *
   * The specified logger factory object is installed as the current
   * factory and the old factory is returned. This is useful for
   * setting up loggers that are tightly coupled with the application.
   *
   * @param fact Pointer to new factory.
   */
  void set_logger_factory( std::unique_ptr< logger_ns::kwiver_logger_factory >&& fact );

private:
  friend VITAL_LOGGER_EXPORT logger_handle_t
    get_logger( const char * const name );

  kwiver_logger_manager();
  void load_factory( std::string const& lib_name );

  class impl;
  const std::unique_ptr< impl > m_impl;

  static kwiver_logger_manager * s_instance;
}; // end class kwiver_logger_manager

} } // end namespace kwiver

#endif
