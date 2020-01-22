/*ckwg +30
 * Copyright 2019-2020 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

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
