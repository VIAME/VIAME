/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Selectively filter input frames
 */

#include "filter_frame_process.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/vital_types.h>

#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

#include <sstream>
#include <iostream>
#include <list>
#include <limits>
#include <cmath>


namespace viame
{

namespace core
{

create_config_trait( detection_threshold, double, "0.0",
  "Require having a detection with at least this confidence to pass frame" );

//------------------------------------------------------------------------------
// Private implementation class
class filter_frame_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  double m_detection_threshold;
};

// =============================================================================

filter_frame_process
::filter_frame_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new filter_frame_process::priv() )
{
  make_ports();
  make_config();
}


filter_frame_process
::~filter_frame_process()
{
}


// -----------------------------------------------------------------------------
void
filter_frame_process
::_configure()
{
  d->m_detection_threshold = config_value_using_trait( detection_threshold );
}


// -----------------------------------------------------------------------------
void
filter_frame_process
::_step()
{
  kwiver::vital::image_container_sptr image;
  kwiver::vital::detected_object_set_sptr detections;

  image = grab_from_port_using_trait( image );

  if( has_input_port_edge_using_trait( detected_object_set ) )
  {
    detections = grab_from_port_using_trait( detected_object_set );
  }

  bool criteria_met = false;

  if( detections )
  {
    for( auto detection : *detections )
    {
      if( detection->confidence() >= d->m_detection_threshold )
      {
        criteria_met = true;
        break;
      }
      else if( detection->type() )
      {
        try
        {
          double score;
          std::string unused;

          detection->type()->get_most_likely( unused, score );

          if( score >= d->m_detection_threshold )
          {
            criteria_met = true;
            break;
          }
        }
        catch( ... )
        {
          continue;
        }
      }
    }
  }

  if( criteria_met )
  {
    push_to_port_using_trait( image, image );
  }
  else
  {
    push_to_port_using_trait( image, kwiver::vital::image_container_sptr() );
  }
}


// -----------------------------------------------------------------------------
void
filter_frame_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( detected_object_set, optional );

  // -- output --
  declare_output_port_using_trait( image, optional );
}


// -----------------------------------------------------------------------------
void
filter_frame_process
::make_config()
{
  declare_config_using_trait( detection_threshold );
}


// =============================================================================
filter_frame_process::priv
::priv()
  : m_detection_threshold( 0.0 )
{
}


filter_frame_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
