// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation and template instantiation for
/// \link kwiver::vital::pointcloud pointcloud \endlink objects

#include "pointcloud.h"
#include <viame/algorithm_framework/io/eigen_io.h>

namespace kwiver {

namespace vital {

/// Default Constructor
template < typename T >
pointcloud_< T >

::pointcloud_()
{}

/// Constructor for a pointcloud
template < typename T >
pointcloud_< T >

::pointcloud_( std::vector< vector_< 3, T > > const& pos )
  : pos_( pos )
{}

/// Constructor for a pointcloud_ from a pointcloud
template < typename T >
pointcloud_< T >

::pointcloud_( pointcloud const& pc )
  : colors_( pc.colors() )
{
  for( auto p : pc.positions() )
  {
    pos_.push_back( p.cast< T >() );
  }
}

/// Accessor for the world coordinates
template < typename T >
std::vector< vector_3d >
pointcloud_< T >
::positions() const
{
  std::vector< vector_3d > retval;
  for( auto p : pos_ )
  {
    retval.push_back( p.template cast< double >() );
  }
  return retval;
}

/// Set the point cloud positions
template < typename T >
void
pointcloud_< T >
::set_positions( std::vector< vector_< 3, T > > const& pos )
{
  pos_.assign( pos.begin(), pos.end() );
}

/// Set the RGB colors of the point cloud
template < typename T >
void
pointcloud_< T >
::set_color( std::vector< rgb_color > const& colors )
{
  colors_.assign( colors.begin(), colors.end() );
}

/// Set the intensities of the point cloud
template < typename T >
void
pointcloud_< T >
::set_intensities( std::vector< uint8_t > const& inten )
{
  inten_.assign( inten.begin(), inten.end() );
}

/// \cond DoxygenSuppress
#define INSTANTIATE_POINTCLOUD( \
  T ) template class VITAL_TYPES_EXPORT pointcloud_< T >;

INSTANTIATE_POINTCLOUD( double );
INSTANTIATE_POINTCLOUD( float );

#undef INSTANTIATE_POINTCLOUD
/// \endcond

} // namespace vital

}   // end namespace vital
