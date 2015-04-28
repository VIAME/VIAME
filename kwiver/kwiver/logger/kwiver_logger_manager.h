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

#ifndef KWIVER_KWIVER_LOGGER_MANAGER_H_
#define KWIVER_KWIVER_LOGGER_MANAGER_H_

#include "kwiver_logger.h"

#include <string>
#include <boost/noncopyable.hpp>
#include <boost/scoped_ptr.hpp>

namespace kwiver {
namespace logger_ns {
  class kwiver_logger_factory;
}

// ----------------------------------------------------------------
/** Logger manager (root object)
 *
 * This class represents the main top level logic for the KWIVER
 * logger. Only one object of this type is required, so this is a
 * singleton created by the static instance() method.
 */
class kwiver_logger_manager
  :private boost::noncopyable
{
public:
  virtual ~kwver_logger_manager();

  /** Get the single instance of this class. */
  static kwiver_logger_manager * instance();

  /**
   * @brief Get name of current logger factory.
   *
   * @return Name of logger factory.
   */
  std::string const&  get_factory_name() const;

private:
  friend logger_handle_t get_logger( const char * const name );

  logger_manager();
  void load_factory( std::string const& lib_name );

  boost::scoped_ptr< logger_ns::kwiver_logger_factory > m_logFactory;
  vidtksys::DynamicLoader::LibraryHandle m_libHandle;

  static kwiver_logger_manager * s_instance;
}; // end class kwiver_logger_manager

} // end namespace kwiver

#endif /* KWIVER_KWIVER_LOGGER_MANAGER_H_ */
