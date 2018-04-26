/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief core feature implementation
 */

#include "feature.h"
#include <vital/io/eigen_io.h>

namespace kwiver {
namespace vital {

/// output stream operator for a feature base class
std::ostream&
operator<<( std::ostream& s, feature const& f )
{
  // TODO include covariance once stream operators are defined
  s << f.loc().transpose() << " "
    << f.magnitude() << " "
    << f.scale() << " "
    << f.angle() << " "
    << f.color();
  return s;
}


/// Default Constructor
template < typename T >
feature_< T >
::feature_()
  : loc_( 0, 0 ),
  magnitude_( 0 ),
  scale_( 1 ),
  angle_( 0 )
{
}


/// Constructor for a feature
template < typename T >
feature_< T >
::feature_( Eigen::Matrix< T, 2, 1 > const& loc,
            T mag, T scale, T angle,
            rgb_color const& color)
  : loc_( loc ),
  magnitude_( mag ),
  scale_( scale ),
  angle_( angle ),
  color_( color )
{
}


/// Constructor for a feature_ from a feature
template < typename T >
feature_< T >
::feature_( feature const& f )
  : loc_( f.loc().cast< T > () ),
  magnitude_( static_cast< T > ( f.magnitude() ) ),
  scale_( static_cast< T > ( f.scale() ) ),
  angle_( static_cast< T > ( f.angle() ) ),
  color_( f.color() )
{
}

template < typename T >
feature_sptr
feature_< T >
::clone() const {
  return std::make_shared<feature_<T>>(this->loc_, this->magnitude_, this->scale_, this->angle_, this->color_);
}


/// output stream operator for a feature
template < typename T >
std::ostream&
operator<<( std::ostream& s, feature_< T > const& f )
{
  // TODO include covariance once stream operators are defined
  s << f.get_loc().transpose() << " "
    << f.get_magnitude() << " "
    << f.get_scale() << " "
    << f.get_angle() << " "
    << f.get_color();
  return s;
}


/// input stream operator for a feature
template < typename T >
std::istream&
operator>>( std::istream& s, feature_< T >& f )
{
  // TODO include covariance once stream operators are defined
  Eigen::Matrix< T, 2, 1 > loc;
  T magnitude;
  T scale;
  T angle;
  rgb_color color;

  s >> loc
    >> magnitude
    >> scale
    >> angle
    >> color;
  f.set_loc( loc );
  f.set_magnitude( magnitude );
  f.set_scale( scale );
  f.set_angle( angle );
  f.set_color( color );
  return s;
}


/// \cond DoxygenSuppress
#define INSTANTIATE_FEATURE( T )                     \
  template class VITAL_EXPORT feature_< T >;         \
  template VITAL_EXPORT std::ostream&                    \
  operator<<( std::ostream& s, feature_< T > const& f ); \
  template VITAL_EXPORT std::istream&                    \
  operator>>( std::istream& s, feature_< T >& f )

INSTANTIATE_FEATURE( double );
INSTANTIATE_FEATURE( float );

#undef INSTANTIATE_FEATURE
/// \endcond

} } // end namespace vital
