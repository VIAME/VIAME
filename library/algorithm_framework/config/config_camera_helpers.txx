// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef CONFIG_CAMERA_HELPERS_TXX
#define CONFIG_CAMERA_HELPERS_TXX

#include <viame/algorithm_framework/config/config_helpers.txx>
#include <viame/core_types/camera_intrinsics.h>
#include <viame/core_types/vector.h>

namespace kwiver::vital {

//// ----------------------------------------------------------------------------
/// Helper function for properly setting a nested camera_intrinsics's
/// configuration
///
/// If the value for the config parameter "type" is supported by the
/// concrete camera_intrinsics class, then a new camera_intrinsics object is
/// created,
/// configured using the set_configuration() method and returned via
/// the \c nested_camera_intrinsics pointer.
///
/// The nested camera_intrinsics will not be set if the implementation type (as
/// defined in the \c get_nested_camera_intrinsics_configuration) is not present
/// or set to
/// an invalid value relative to the registered names for this
/// \c type_name
///
/// \tparam    INTERFACE           interface that the nested_camera_intrinsics
/// implements.
/// \param[in] name                Config block name for the nested
/// camera_intrinsics.
/// \param[in] config              The \c config_block instance from which we
/// will
///                                draw configuration needed for the nested
///                                camera_intrinsics instance.
/// \param[out] nested_camera_intrinsics The nested camera_intrinsics's sptr
/// variable.
inline void
set_nested_camera_intrinsics_configuration(
  std::string const& name,
  config_block_sptr config,
  camera_intrinsics_sptr&    K )
{
  vital::config_block_sptr bc = config->subblock( "base_camera" );

  simple_camera_intrinsics K2( bc->get_value< double >(
    "focal_length",
    K->focal_length() ),
    bc->get_value< vector_2d >(
      "principal_point",
      K->principal_point() ),
    bc->get_value< double >(
      "aspect_ratio",
      K->aspect_ratio() ),
    bc->get_value< double >(
      "skew",
      K->skew() ) );
  double r1 = bc->get_value< double >( "r1", 0 );
  double r2 = bc->get_value< double >( "r2", 0 );
  double r3 = bc->get_value< double >( "r3", 0 );

  Eigen::VectorXd dist;
  dist.resize( 5 );
  dist.setZero();
  dist[ 0 ] = r1;
  dist[ 1 ] = r2;
  dist[ 4 ] = r3;
  K2.set_dist_coeffs( dist );
  K = K2.clone();
}

//// ----------------------------------------------------------------------------
/// Helper function for properly getting a nested camera_intrinsics's
/// configuration
///
/// Adds a configurable camera_intrinsics implementation switch for this
/// camera_intrinsics.
/// If the variable pointed to by \c nested_camera_intrinsics is a defined sptr
/// to an
/// implementation, its \link kwiver::vital::config_block configuration
/// \endlink
/// parameters are merged with the given
/// \link kwiver::vital::config_block config_block \endlink.
///
/// \tparam          INTERFACE   interface that the nested_camera_intrinsics
/// implements.
/// \param[in]       name        An identifying name for the nested
/// camera_intrinsics
/// \param[in,out]   config      The \c config_block instance in which to put
/// the
///                              nested camera_intrinsics's configuration.
/// \param[in]       nested_camera_intrinsics The nested camera_intrinsics's
/// sptr variable.
inline void
get_nested_camera_intrinsics_configuration(
  std::string const& name,
  config_block_sptr config,
  camera_intrinsics_sptr K )
{
  if( K )
  {
    auto bc = config->subblock_view( name );

    bc->set_value(
      "focal_length", K->focal_length(),
      "focal length of the base camera model" );

    bc->set_value(
      "principal_point", K->principal_point().transpose(),
      "The principal point of the base camera model \"x y\".\n"
      "It is usually safe to assume this is the center of the "
      "image." );

    bc->set_value(
      "aspect_ratio", K->aspect_ratio(),
      "the pixel aspect ratio of the base camera model" );

    bc->set_value(
      "skew", K->skew(),
      "The skew factor of the base camera model.\n"
      "This is almost always zero in any real camera." );

    double r1 = 0;
    double r2 = 0;
    double r3 = 0;
    auto dc = K->dist_coeffs();
    if( dc.size() == 5 )
    {
      r1 = dc[ 0 ];
      r2 = dc[ 1 ];
      r3 = dc[ 4 ];
    }
    bc->set_value( "r1", r1, "r^2 radial distortion term" );
    bc->set_value( "r2", r2, "r^4 radial distortion term" );
    bc->set_value( "r3", r3, "r^6 radial distortion term" );
  }
}

// ----------------------------------------------------------------------------
// specializations of set/get_config_helper for shared_ptr< camera_intrinsics >
// for base implementations see config_helpers.txx

// A helper for populating \p key in \p config based on the  configuration of
// the camera_intrinsics given in \p value.
template < typename ValueType,
  typename std::enable_if_t< detail::is_shared_ptr< ValueType >::value,
    bool > = true,
  typename std::enable_if_t< std::is_base_of_v< kwiver::vital::camera_intrinsics,
    typename ValueType::element_type >, bool > = true >
void
set_config_helper(
  config_block_sptr config, const std::string& key,
  const ValueType& value,
  [[maybe_unused]] config_block_description_t const& description  =
  config_block_description_t() )
{
  if( value )
  {
    ValueType cam_intrinsics = value->clone();
    kwiver::vital::set_nested_camera_intrinsics_configuration(
      key, config,
      cam_intrinsics );
  }

  // We only set a value to assign a description to the key.
  // The value will never be read from the config itself,
  // as this type has a custom specialized accessor
  // that returns a new instance each time.
  config->set_value< ValueType >( key, value, description );
}

// A helper for getting a value from a config block. This specialization is for
// keys that correspond to nested camera_intrinsics.
template < typename ValueType,
  typename std::enable_if_t< detail::is_shared_ptr< ValueType >::value,
    bool > = true,
  typename std::enable_if_t< std::is_base_of_v< kwiver::vital::camera_intrinsics,
    typename ValueType::element_type >, bool > = true >
ValueType
get_config_helper( config_block_sptr config, config_block_key_t const& key )
{
  ValueType cam_intrinsics;
  kwiver::vital::get_nested_camera_intrinsics_configuration(
    key, config,
    cam_intrinsics );
  return cam_intrinsics;
}

} // namespace kwiver::vital

#endif
