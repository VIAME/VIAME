/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation for full_frame_detector
 */

#include "full_frame_detector.h"

#include <vector>


namespace viame {

namespace kv = kwiver::vital;

/// Private implementation class
class full_frame_detector::priv
{
public:

  /// Constructor
  priv()
  : detection_type( "generic_object_proposal" )
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  std::string detection_type;
};


/// Constructor
full_frame_detector
::full_frame_detector()
  : d( new priv )
{
}


full_frame_detector
::~full_frame_detector()
{
}


/// Settings
kv::config_block_sptr
full_frame_detector
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::image_object_detector::get_configuration();

  config->set_value( "detection_type", d->detection_type,
                     "Object type to add to newly created detected objects" );

  return config;
}


void
full_frame_detector
::set_configuration(kv::config_block_sptr config)
{
  d->detection_type = config->get_value<std::string>( "detection_type" );
}


bool
full_frame_detector
::check_configuration(kv::config_block_sptr config) const
{
  return true;
}


/// Run full frame descriptor
kv::detected_object_set_sptr
full_frame_detector
::detect( kv::image_container_sptr image_data ) const
{
  auto output = std::make_shared< kv::detected_object_set >();

  if( image_data->height() > 0 && image_data->width() > 0 )
  {
    kv::bounding_box_d det_box( 0, 0,
                                image_data->width(),
                                image_data->height() );

    if( !d->detection_type.empty() )
    {
      auto dot = std::make_shared< kv::detected_object_type >();
      dot->set_score( d->detection_type, 1.0 );

      output->add(
        std::make_shared< kv::detected_object >(
          det_box, 1.0, dot ) );
    }
    else
    {
      output->add(
        std::make_shared< kv::detected_object >(
          det_box, 1.0 ) );
    }
  }

  return output;
}

} // end namespace viame
