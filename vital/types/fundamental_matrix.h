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

/**
 * \file
 * \brief Core fundamental matrix class definition
 */

#ifndef VITAL_FUNDAMENTAL_MATRIX_H_
#define VITAL_FUNDAMENTAL_MATRIX_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vital/types/matrix.h>
#include <vital/types/vector.h>

#include <iostream>
#include <map>
#include <vector>
#include <memory>

namespace kwiver {
namespace vital {


// Forward declarations of abstract fundamental matrix class
class fundamental_matrix;
// typedef for a fundamental_matrix shared pointer
typedef std::shared_ptr< fundamental_matrix > fundamental_matrix_sptr;


// ===========================================================================
// Fundmental Matrix Base-class
// ---------------------------------------------------------------------------

/// Abstract base fundamental matrix representation class
class VITAL_EXPORT fundamental_matrix
{
public:
  /// Destructor
  virtual ~fundamental_matrix() = default;

  /// Create a clone of this fundamental_matrix object, returning as smart pointer
  /**
   * \return A new deep clone of this fundamental_matrix
   */
  virtual fundamental_matrix_sptr clone() const = 0;

  /// Get a double-typed copy of the underlying matrix
  /**
   * \return A copy of the matrix represented in the double type.
   */
  virtual matrix_3x3d matrix() const = 0;
};


// ===========================================================================
// Typed Fundmental Matrix
// ---------------------------------------------------------------------------

/// Representation of a templated Eigen-based fundamental matrix
template <typename T>
class VITAL_EXPORT fundamental_matrix_
  : public fundamental_matrix
{
public:
  typedef T value_type;
  typedef Eigen::Matrix<T,3,3> matrix_t;
  typedef Eigen::Matrix<T,3,1> vector_t;

  /// Construct from a provided matrix by projection.
  /** Decompose and find closest fundamental matrix to the input \p mat.
   * \param mat The 3x3 matrix to use.
   */
  explicit
  fundamental_matrix_<T>( matrix_t const &mat );

  /// Conversion Copy constructor
  /**
   * \param other The other fundamental_matrix to be copied.
   */
  template <typename U>
  explicit
  fundamental_matrix_<T>( fundamental_matrix_<U> const &other )
    : mat_( other.mat_.template cast<T>() )
  {
  }

  /// Construct from a generic fundamental_matrix
  explicit
  fundamental_matrix_<T>( fundamental_matrix const &base );

  // Abstract method definitions ---------------------------------------------

  /// Create a clone of ourself as a shared pointer
  /**
   * \return A new clone of this fundamental_matrix.
   */
  virtual fundamental_matrix_sptr clone() const;

  /// Get a double-typed copy of the underlying matrix
  /**
   * \return A copy of the matrix represented in the double type.
   */
  virtual Eigen::Matrix<double,3,3> matrix() const;


  // Member Functions --------------------------------------------------------

protected:
  /// the 3x3 matrix used to represent the fundamental matrix
  matrix_t mat_;
};


/// Double-precision camera type
typedef fundamental_matrix_<double> fundamental_matrix_d;
/// Single-precision camera type
typedef fundamental_matrix_<float> fundamental_matrix_f;


// ===========================================================================
// Utility Functions
// ---------------------------------------------------------------------------

/// Output stream operator for \p fundamental_matrix base-class
VITAL_EXPORT std::ostream& operator<<( std::ostream &s,
                                       fundamental_matrix const &f );


} } // end namespace vital

#endif // VITAL_FUNDAMENTAL_MATRIX_H_
