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
 * \brief core fundamental matrix template implementations
 */

#include "fundamental_matrix.h"

#include <cmath>

#include <vital/exceptions/math.h>

#include <Eigen/SVD>


namespace kwiver {
namespace vital {


/// Construct from a provided matrix
template <typename T>
fundamental_matrix_<T>
::fundamental_matrix_( Eigen::Matrix<T,3,3> const &mat )
{
  Eigen::JacobiSVD<matrix_t> svd(mat, Eigen::ComputeFullU |
                                      Eigen::ComputeFullV);
  auto S = svd.singularValues();
  const matrix_t& U = svd.matrixU();
  const matrix_t& V = svd.matrixV();

  // clear the last singular value
  S[2] = T(0);
  S /= S.norm();
  mat_ = U*S.asDiagonal()*V.transpose();
}

/// Conversion Copy constructor -- float specialization
template <>
template <>
fundamental_matrix_<float>
::fundamental_matrix_( fundamental_matrix_<float> const &other )
  : mat_( other.mat_ )
{
}

/// Conversion Copy constructor -- double specialization
template <>
template <>
fundamental_matrix_<double>
::fundamental_matrix_( fundamental_matrix_<double> const &other )
  : mat_( other.mat_ )
{
}

/// Construct from a generic fundamental_matrix
template <typename T>
fundamental_matrix_<T>
::fundamental_matrix_( fundamental_matrix const &base )
  : mat_( base.matrix().template cast<T>() )
{
}

/// Construct from a generic fundamental_matrix -- double specialization
template <>
fundamental_matrix_<double>
::fundamental_matrix_( fundamental_matrix const &base )
  : mat_( base.matrix() )
{
}

/// Create a clone of outself as a shared pointer
template <typename T>
fundamental_matrix_sptr
fundamental_matrix_<T>
::clone() const
{
  return fundamental_matrix_sptr( new fundamental_matrix_<T>( *this ) );
}

/// Get a double-typed copy of the underlying matrix
template <typename T>
Eigen::Matrix<double,3,3>
fundamental_matrix_<T>
::matrix() const
{
  return this->mat_.template cast<double>();
}

/// Specialization for matrices with native double type
template <>
Eigen::Matrix<double,3,3>
fundamental_matrix_<double>
::matrix() const
{
  return this->mat_;
}


// ===========================================================================
// Other Functions
// ---------------------------------------------------------------------------

/// Output stream operator for \p fundamental_matrix instances
std::ostream&
operator<<( std::ostream &s, fundamental_matrix const &f )
{
  s << f.matrix();
  return s;
}

// ===========================================================================
// Template class instantiation
// ---------------------------------------------------------------------------
/// \cond DoxygenSuppress
#define INSTANTIATE_FUNDAMENTAL_MATRIX(T) \
  template class fundamental_matrix_<T>;

INSTANTIATE_FUNDAMENTAL_MATRIX(float);
INSTANTIATE_FUNDAMENTAL_MATRIX(double);
#undef INSTANTIATE_FUNDAMENTAL_MATRIX
/// \endcond


} } // end vital namespace
