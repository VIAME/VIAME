// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Core transform class definition
 */

#ifndef VITAL_TRANSFORM_2D_H_
#define VITAL_TRANSFORM_2D_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vital/types/vector.h>

#include <memory>

namespace kwiver {
namespace vital {

// Forward declarations of abstract transform class
class transform_2d;
// typedef for a transform shared pointer
typedef std::shared_ptr< transform_2d > transform_2d_sptr;

// ============================================================================
/// Abstract base transformation representation class
class VITAL_EXPORT transform_2d
{
public:
  /// Destructor
  virtual ~transform_2d() = default;

  /// Create a clone of this transform object, returning as smart pointer
  /**
   * \return A new deep clone of this transformation.
   */
  virtual transform_2d_sptr clone() const = 0;

  /// Map a 2D double-type point using this transform
  /**
   * \param p Point to map against this transform
   * \return New point in the projected coordinate system.
   */
  virtual vector_2d map( vector_2d const& p ) const = 0;

  /// Return an inverse of this transform object
  /**
   * \throws non_invertible
   *   When the transformation is non-invertible.
   * \return A new transform object that is the inverse of this transformation.
   */
  transform_2d_sptr inverse() const { return this->inverse_(); }

protected:
  virtual transform_2d_sptr inverse_() const = 0;
};

} // namespace vital
} // namespace kwiver

#endif
