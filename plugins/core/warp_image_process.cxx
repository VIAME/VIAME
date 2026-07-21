/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Warp an image using a 2D homography loaded from a file
 */

#include "warp_image_process.h"

#include <vital/vital_types.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/algo/transform_2d_io.h>
#include <vital/types/homography.h>
#include <vital/types/image_container.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdexcept>

namespace viame
{

namespace core
{

create_config_trait( transformation_file, kwiver::vital::path_t, "",
  "File containing the 2D homography mapping this image's coordinates "
  "into the target camera's. Read with the transform_reader algorithm "
  "(default type \"auto\": DIVE camera registration .json or plain text "
  "3x3 homography)." );
create_config_trait( inverse, bool, "false",
  "Apply the inverse of the loaded transform instead" );

create_port_trait( size_image, image, "Image to get output size from." );

//------------------------------------------------------------------------------
// Private implementation class
class warp_image_process::priv
{
public:
  priv() {}
  ~priv() {}

  // Configuration values
  kwiver::vital::path_t m_transformation_file;
  bool m_inverse = false;
  cv::Mat m_warp;
};

// =============================================================================

warp_image_process
::warp_image_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new warp_image_process::priv() )
{
  make_ports();
  make_config();
}


warp_image_process
::~warp_image_process()
{
}


// -----------------------------------------------------------------------------
void
warp_image_process
::_configure()
{
  d->m_transformation_file = config_value_using_trait( transformation_file );
  d->m_inverse = config_value_using_trait( inverse );

  if( d->m_transformation_file.empty() )
  {
    throw std::runtime_error( "warp_image requires a transformation_file" );
  }

  kwiver::vital::config_block_sptr algo_config = get_config();

  if( !algo_config->has_value( "transform_reader:type" ) )
  {
    algo_config->set_value( "transform_reader:type", "auto" );
  }

  kwiver::vital::algo::transform_2d_io_sptr reader;

  kwiver::vital::algo::transform_2d_io::set_nested_algo_configuration(
    "transform_reader", algo_config, reader );

  if( !reader )
  {
    throw std::runtime_error( "Unable to create transform_reader" );
  }

  kwiver::vital::transform_2d_sptr transform =
    reader->load( d->m_transformation_file );

  if( d->m_inverse )
  {
    transform = transform->inverse();
  }

  // Image warping needs the full 3x3 matrix, not just point mapping, so
  // only homography transforms (DIVE .json, plain text) are supported.
  auto homog = std::dynamic_pointer_cast< kwiver::vital::homography >( transform );

  if( !homog )
  {
    throw std::runtime_error(
      "warp_image requires a homography transform: " + d->m_transformation_file );
  }

  Eigen::Matrix< double, 3, 3 > const matrix = homog->matrix();
  cv::eigen2cv( matrix, d->m_warp );
}


// -----------------------------------------------------------------------------
void
warp_image_process
::_step()
{
  kwiver::vital::image_container_sptr image, size_image;

  image = grab_from_port_using_trait( image );

  cv::Size output_size( image->width(), image->height() );

  if( has_input_port_edge_using_trait( size_image ) )
  {
    size_image = grab_from_port_using_trait( size_image );

    output_size = cv::Size( size_image->width(), size_image->height() );
  }

  try
  {
    cv::Mat input = kwiver::arrows::ocv::image_container::vital_to_ocv(
      image->get_image(),
      kwiver::arrows::ocv::image_container::BGR_COLOR );

    cv::Mat output;
    cv::warpPerspective( input, output, d->m_warp, output_size );

    push_to_port_using_trait( image,
      kwiver::vital::image_container_sptr(
        new kwiver::arrows::ocv::image_container( output,
          kwiver::arrows::ocv::image_container::BGR_COLOR ) ) );
  }
  catch( ... )
  {
    push_to_port_using_trait( image, kwiver::vital::image_container_sptr() );
  }
}


// -----------------------------------------------------------------------------
void
warp_image_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( size_image, optional );

  // -- output --
  declare_output_port_using_trait( image, optional );
}


// -----------------------------------------------------------------------------
void
warp_image_process
::make_config()
{
  declare_config_using_trait( transformation_file );
  declare_config_using_trait( inverse );
}

} // end namespace core

} // end namespace viame
