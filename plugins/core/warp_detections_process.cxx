/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Warp detections using a 2D transform loaded from a file
 */

#include "warp_detections_process.h"

#include <vital/vital_types.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/algo/transform_2d_io.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/transform_2d.h>

#include <algorithm>
#include <stdexcept>

namespace viame
{

namespace core
{

create_config_trait( transformation_file, kwiver::vital::path_t, "",
  "File containing the 2D transform mapping this camera's image "
  "coordinates into the target camera's. Read with the transform_reader "
  "algorithm (default type \"auto\": DIVE camera registration .json or "
  "plain text 3x3 homography)." );
create_config_trait( inverse, bool, "false",
  "Apply the inverse of the loaded transform instead" );

//------------------------------------------------------------------------------
// Private implementation class
class warp_detections_process::priv
{
public:
  priv() {}
  ~priv() {}

  // Configuration values
  kwiver::vital::path_t m_transformation_file;
  bool m_inverse = false;
  kwiver::vital::transform_2d_sptr m_transform;
};

// =============================================================================

warp_detections_process
::warp_detections_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new warp_detections_process::priv() )
{
  make_ports();
  make_config();
}


warp_detections_process
::~warp_detections_process()
{
}


// -----------------------------------------------------------------------------
void
warp_detections_process
::_configure()
{
  d->m_transformation_file = config_value_using_trait( transformation_file );
  d->m_inverse = config_value_using_trait( inverse );

  if( d->m_transformation_file.empty() )
  {
    throw std::runtime_error( "warp_detections requires a transformation_file" );
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

  d->m_transform = reader->load( d->m_transformation_file );

  if( d->m_inverse )
  {
    d->m_transform = d->m_transform->inverse();
  }
}


// -----------------------------------------------------------------------------
void
warp_detections_process
::_step()
{
  kwiver::vital::detected_object_set_sptr input;
  kwiver::vital::detected_object_set_sptr output;

  input = grab_from_port_using_trait( detected_object_set );

  try
  {
    if( input )
    {
      output = input->clone();

      for( auto detection : *output )
      {
        auto const& bbox = detection->bounding_box();

        kwiver::vital::vector_2d const corners[4] = {
          d->m_transform->map( { bbox.min_x(), bbox.min_y() } ),
          d->m_transform->map( { bbox.max_x(), bbox.min_y() } ),
          d->m_transform->map( { bbox.max_x(), bbox.max_y() } ),
          d->m_transform->map( { bbox.min_x(), bbox.max_y() } ) };

        double min_x = corners[0][0], max_x = corners[0][0];
        double min_y = corners[0][1], max_y = corners[0][1];

        for( unsigned i = 1; i < 4; ++i )
        {
          min_x = std::min( min_x, corners[i][0] );
          max_x = std::max( max_x, corners[i][0] );
          min_y = std::min( min_y, corners[i][1] );
          max_y = std::max( max_y, corners[i][1] );
        }

        detection->set_bounding_box(
          kwiver::vital::bounding_box_d( min_x, min_y, max_x, max_y ) );
      }
    }
  }
  catch( ... )
  {
    push_to_port_using_trait( detected_object_set,
      kwiver::vital::detected_object_set_sptr() );
    return;
  }

  push_to_port_using_trait( detected_object_set, output );
}


// -----------------------------------------------------------------------------
void
warp_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, required );
}


// -----------------------------------------------------------------------------
void
warp_detections_process
::make_config()
{
  declare_config_using_trait( transformation_file );
  declare_config_using_trait( inverse );
}

} // end namespace core

} // end namespace viame
