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
#include <vital/algo/warp_image.h>
#include <vital/algo/algorithm.txx>
#include <vital/types/homography.h>
#include <vital/types/image_container.h>

#include <cstring>
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

namespace
{

kwiver::vital::image_container_sptr
blank_canvas( const kwiver::vital::image& source, size_t width, size_t height )
{
  kwiver::vital::image output(
    width, height, source.depth(), true, source.pixel_traits() );

  std::memset( output.memory()->data(), 0, output.memory()->size() );

  return std::make_shared< kwiver::vital::simple_image_container >( output );
}

} // end anonymous namespace

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
  kwiver::vital::homography_sptr m_homography;
  kwiver::vital::algo::warp_image_sptr m_warper;
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

  if( !algo_config->has_value( "warper:type" ) )
  {
    algo_config->set_value( "warper:type", "ocv" );
  }

  kwiver::vital::algo::transform_2d_io_sptr reader;

  kwiver::vital::set_nested_algo_configuration<
    kwiver::vital::algo::transform_2d_io >(
    "transform_reader", algo_config, reader );

  if( !reader )
  {
    throw std::runtime_error( "Unable to create transform_reader" );
  }

  kwiver::vital::set_nested_algo_configuration<
    kwiver::vital::algo::warp_image >(
    "warper", algo_config, d->m_warper );

  if( !d->m_warper )
  {
    throw std::runtime_error( "Unable to create warper" );
  }

  kwiver::vital::transform_2d_sptr transform =
    reader->load( d->m_transformation_file );

  if( d->m_inverse )
  {
    transform = transform->inverse();
  }

  // Image warping needs the full 3x3 matrix, not just point mapping, so
  // only homography transforms (DIVE .json, plain text) are supported.
  d->m_homography =
    std::dynamic_pointer_cast< kwiver::vital::homography >( transform );

  if( !d->m_homography )
  {
    throw std::runtime_error(
      "warp_image requires a homography transform: " + d->m_transformation_file );
  }
}


// -----------------------------------------------------------------------------
void
warp_image_process
::_step()
{
  kwiver::vital::image_container_sptr image, size_image;

  image = grab_from_port_using_trait( image );

  size_t output_width = image->width();
  size_t output_height = image->height();

  if( has_input_port_edge_using_trait( size_image ) )
  {
    size_image = grab_from_port_using_trait( size_image );

    output_width = size_image->width();
    output_height = size_image->height();
  }

  try
  {
    push_to_port_using_trait( image,
      d->m_warper->warp( image,
        blank_canvas( image->get_image(), output_width, output_height ),
        d->m_homography ) );
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
