/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "refine_detections_add_fixed.h"

namespace viame {

namespace kv = kwiver::vital;

/// Private implementation class
class refine_detections_add_fixed::priv
{
public:

  /// Constructor
  priv()
  : add_full_image_detection( true )
  , detection_type( "generic_object_proposal" )
  {
  }

  /// Destructor
  ~priv()
  {
  }

  /// Parameters
  bool add_full_image_detection;
  std::string detection_type;
};


/// Constructor
refine_detections_add_fixed
::refine_detections_add_fixed()
: d_( new priv() )
{
}


/// Destructor
refine_detections_add_fixed
::~refine_detections_add_fixed()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
kv::config_block_sptr
refine_detections_add_fixed
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::refine_detections::get_configuration();

  config->set_value( "add_full_image_detection", d_->add_full_image_detection,
                     "Add full image detection of the same size as the input image." );
  config->set_value( "detection_type", d_->detection_type,
                     "Object type to add to newly created detected objects" );

  return config;
}


/// Set this algorithm's properties via a config block
void
refine_detections_add_fixed
::set_configuration( kv::config_block_sptr in_config )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  d_->add_full_image_detection = config->get_value<bool>( "add_full_image_detection" );
  d_->detection_type = config->get_value<std::string>( "detection_type" );
}

/// Check that the algorithm's currently configuration is valid
bool
refine_detections_add_fixed
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------------
kv::detected_object_set_sptr
refine_detections_add_fixed
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  kv::detected_object_set_sptr output = detections
    ? detections->clone()
    : std::make_shared< kv::detected_object_set >();

  if( d_->add_full_image_detection && image_data &&
      image_data->height() > 0 && image_data->width() > 0 )
  {
    kv::bounding_box_d det_box( 0, 0,
                                image_data->width(),
                                image_data->height() );

    if( !d_->detection_type.empty() )
    {
      auto dot = std::make_shared< kv::detected_object_type >();
      dot->set_score( d_->detection_type, 1.0 );

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
