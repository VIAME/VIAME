/*ckwg +29
 * Copyright 2014-2020 by Kitware, Inc.
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
 * \brief VITAL Exceptions pertaining to math operations.
 */

#ifndef VITAL_CORE_EXCEPTIONS_MATH_H
#define VITAL_CORE_EXCEPTIONS_MATH_H

#include "base.h"
#include <string>
#include <vital/vital_types.h>

namespace kwiver {
namespace vital {


/// VITAL Generic math exception
class VITAL_EXCEPTIONS_EXPORT math_exception
  : public vital_exception
{
public:
  /// Constructor
  math_exception() noexcept;
  /// Destructor
  virtual ~math_exception() noexcept;
};


/// Exception for when an instance of a conceptually invertible object is
/// non-invertible
class VITAL_EXCEPTIONS_EXPORT non_invertible
  : public math_exception
{
public:
  /// Constructor
  non_invertible() noexcept;
  /// Destructor
  virtual ~non_invertible() noexcept;
};


/// Exception for when some point maps to infinity
class VITAL_EXCEPTIONS_EXPORT point_maps_to_infinity
  : public math_exception
{
public:
  /// Constructor
  point_maps_to_infinity() noexcept;
  /// Destructor
  virtual ~point_maps_to_infinity() noexcept;
};


/// We cannot perfom some operation on a matrix
class VITAL_EXCEPTIONS_EXPORT invalid_matrix_operation
  : public math_exception
{
public:
  /// Constructor
  /*
   * \param reason  The reason for invalidity.
   */
  invalid_matrix_operation(std::string reason) noexcept;
  /// Destructor
  virtual ~invalid_matrix_operation() noexcept;

  /// Reason the operation is invalid
  std::string m_reason;
};

} } // end vital namespace

#endif // VITAL_CORE_EXCEPTIONS_MATH_H
