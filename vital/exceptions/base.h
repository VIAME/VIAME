/*ckwg +29
 * Copyright 2013-2016 by Kitware, Inc.
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

/**
 * \file
 * \brief VITAL base exception interface
 */

#ifndef VITAL_CORE_EXCEPTIONS_BASE_H
#define VITAL_CORE_EXCEPTIONS_BASE_H

#include <vital/vital_config.h>
#include <vital/vital_export.h>

#include <string>
#include <exception>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// The base class for all vital exceptions
/**
 * \ingroup exceptions
 */
class VITAL_EXPORT vital_core_base_exception
  : public std::exception
{
public:
  /// Constructor
  vital_core_base_exception() noexcept;

  /// Destructor
  virtual ~vital_core_base_exception() noexcept;

  /**
   * \brief Description of the exception
   * \returns A string describing what went wrong.
   */
  char const* what() const noexcept;

  /**
   * \brief Set optional location of exception.
   *
   * This method saves the supplied source file and location in the
   * exception so the location where the exception was thrown can be
   * determined. This is not that useful for an end user, but it is
   * very helpful for developers.
   *
   * \param file Name of source file.
   * \param line Line number in file.
   */
  void set_location( std::string const& file, int line );
protected:
  /// descriptive string as to what happened to cause the exception.
  std::string m_what;
  std::string m_file_name;
  int m_line_number;
};

// ------------------------------------------------------------------
/// Exception for incorrect input values
/**
 * \ingroup exceptions
 */
class VITAL_EXPORT invalid_value
  : public vital_core_base_exception
{
public:
  /// Constructor
  invalid_value(std::string reason) noexcept;
  /// Destructor
  virtual ~invalid_value() noexcept;
protected:
  /// Reason for invalidity
  std::string m_reason;
};

} } // end namespace vital


// ------------------------------------------------------------------
///Exception helper macro.
/**
 * Macro to simplify creating exception messages using stream
 * operators. The source location of this macro is also recorded in
 * the exception.
 *
 * @param E       Exception type.
 * @param MSG     Stream constructed exception message.
 */
#define VITAL_THROW_MSG(E, MSG) do {            \
    std::stringstream _oss_;                    \
    _oss_ << MSG;                               \
    E except( _oss.str() );                     \
    except.set_location( __file, __line );      \
    throw except;                               \
  } while (0)

#endif // VITAL_CORE_EXCEPTIONS_BASE_H
