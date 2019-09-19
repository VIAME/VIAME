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

#ifndef KWIVER_KWIVER_LOGGER_FACTORY_H_
#define KWIVER_KWIVER_LOGGER_FACTORY_H_

#include "kwiver_logger.h"
#include <vital/noncopyable.h>

namespace kwiver {
namespace vital {
namespace logger_ns {

// ----------------------------------------------------------------
/** Factory for underlying logger.
 *
 * This class is the abstract base class that adapts the KWIVER logger
 * to the underlying logging system.
 *
 * An object of this type can be created early in the program
 * execution (i.e. static initializer time), which is before the
 * initialize method is called.
 */
class VITAL_LOGGER_EXPORT kwiver_logger_factory
  : private kwiver::vital::noncopyable
{
public:
  /**
   * @brief Create logger factory
   *
   * The name for this factory should describe the logger type that is
   * being created.
   *
   * Call get_name() to access this name.
   *
   * @param name  Name of this logger factory
   */
  kwiver_logger_factory( std::string const& name );
  virtual ~kwiver_logger_factory();

  /**
   * @brief Get pointer to named logger.
   *
   * This method returns a pointer to a named logger. The underlying
   * log provider determines what is actually done here.
   *
   * @param name Name of logger object.
   *
   * @return
   */
  virtual logger_handle_t get_logger( std::string const& name ) = 0;

  /**
   * @brief Get logger factory name.
   *
   * Returns the name associated with this logger factory.
   *
   * @return Name or type of logger created by this factory.
   */
  std::string const& get_factory_name() const;

private:
  std::string m_name; // factory name
}; // end class kwiver_logger_factory

} } } // end namespace


#endif
