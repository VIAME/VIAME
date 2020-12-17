// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation and template instantiation for
 * \link kwiver::vital::landmark landmark \endlink objects
 */

#include "landmark.h"
#include <vital/io/eigen_io.h>

namespace kwiver {
namespace vital {

/// output stream operator for a landmark base class
std::ostream&
operator<<( std::ostream& s, landmark const& m )
{
  // TODO include covariance once stream operators are defined
  s << m.loc() << " "
    << m.scale() << " "
    << m.normal() << " "
    << m.color() << " "
    << m.observations() << ""
    << m.cos_obs_angle();
  return s;
}

/// Default Constructor
template < typename T >
landmark_< T >
::landmark_()
  : loc_( 0, 0, 0 ),
    scale_( 1 ),
    normal_( 0, 0, 0 ),
    observations_( 0 ),
    cos_obs_angle_(1.0)
{
}

/// Constructor for a landmark
template < typename T >
landmark_< T >
::landmark_( Eigen::Matrix< T, 3, 1 > const& loc, T scale )
  : loc_( loc ),
    scale_( scale ),
    normal_( 0, 0, 0 ),
    observations_( 0 ),
    cos_obs_angle_(1.0)
{
}

/// Constructor for a landmark_ from a landmark
template < typename T >
landmark_< T >
::landmark_( landmark const& lm )
  : loc_( lm.loc().cast< T > () ),
    scale_( static_cast< T > ( lm.scale() ) ),
    normal_( lm.normal().cast< T > () ),
    color_( lm.color() ),
    observations_( lm.observations() ),
    cos_obs_angle_(static_cast< T >( lm.cos_obs_angle()))
{
}

/// output stream operator for a landmark
template < typename T >
std::ostream&
operator<<( std::ostream& s, landmark_< T > const& m )
{
  // TODO include covariance once stream operators are defined
  s << m.get_loc() << " "
    << m.get_scale() << " "
    << m.get_normal() << " "
    << m.get_color() << " "
    << m.get_observations() << " "
    << m.get_cos_obs_angle();
  return s;
}

/// input stream operator for a landmark
template < typename T >
std::istream&
operator>>( std::istream& s, landmark_< T >& m )
{
  // TODO include covariance once stream operators are defined
  Eigen::Matrix< T, 3, 1 > loc;
  Eigen::Matrix< T, 3, 1 > normal;
  T scale;
  rgb_color color;
  unsigned int observations;
  T cos_observation_angle;

  s >> loc
    >> scale
    >> normal
    >> color
    >> observations
    >> cos_observation_angle;
  m.set_loc( loc );
  m.set_scale( scale );
  m.set_normal( normal );
  m.set_color( color );
  m.set_observations( observations );
  m.set_cos_observation_angle(cos_observation_angle);
  return s;
}

/// \cond DoxygenSuppress
#define INSTANTIATE_LANDMARK( T )                     \
  template class VITAL_EXPORT landmark_< T >;         \
  template VITAL_EXPORT std::ostream&                     \
  operator<<( std::ostream& s, landmark_< T > const& f ); \
  template VITAL_EXPORT std::istream&                     \
  operator>>( std::istream& s, landmark_< T >& f )

INSTANTIATE_LANDMARK( double );
INSTANTIATE_LANDMARK( float );

#undef INSTANTIATE_LANDMARK
/// \endcond

} } // end namespace vital
