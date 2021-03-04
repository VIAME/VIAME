// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
