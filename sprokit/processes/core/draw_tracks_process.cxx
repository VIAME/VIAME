// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Draw tracks process implementation.
 */

#include "draw_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>

#include <vital/algo/draw_tracks.h>

#include <kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

create_port_trait( output_image, image, "Image with tracks" );

// config items
create_algorithm_name_config_trait( draw_tracks );

//----------------------------------------------------------------
// Private implementation class
class draw_tracks_process::priv
{
public:
  priv();
  ~priv();

  algo::draw_tracks_sptr         m_draw_tracks;
};

// ================================================================

draw_tracks_process
::draw_tracks_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new draw_tracks_process::priv )
{
  make_ports();
  make_config();
}

draw_tracks_process
::~draw_tracks_process()
{
}

// ----------------------------------------------------------------
void
draw_tracks_process
::_configure()
{
  scoped_configure_instrumentation();

  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::draw_tracks::set_nested_algo_configuration_using_trait(
    draw_tracks,
    algo_config,
    d->m_draw_tracks );
  if ( ! d->m_draw_tracks )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Unable to create draw_tracks" );
  }

  algo::draw_tracks::get_nested_algo_configuration_using_trait(
    draw_tracks,
    algo_config,
    d->m_draw_tracks );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! algo::draw_tracks::check_nested_algo_configuration_using_trait(
         draw_tracks, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Configuration check failed." );
  }
}

// ----------------------------------------------------------------
void
draw_tracks_process
::_step()
{
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  vital::feature_track_set_sptr tracks = grab_from_port_using_trait( feature_track_set );

  kwiver::vital::image_container_sptr annotated_image;
  {
    scoped_step_instrumentation();

    kwiver::vital::image_container_sptr_list image_list;
    image_list.push_back( img );

    annotated_image = d->m_draw_tracks->draw( tracks, image_list );
  }

  // ( port, value )
  push_to_port_using_trait( output_image, annotated_image );

}

// ----------------------------------------------------------------
void
draw_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( feature_track_set, required );

  // -- output --
  declare_output_port_using_trait( output_image, optional );
}

// ----------------------------------------------------------------
void
draw_tracks_process
::make_config()
{
  declare_config_using_trait( draw_tracks );
}

// ================================================================
draw_tracks_process::priv
::priv()
{
}

draw_tracks_process::priv
::~priv()
{
}

} // end namespace
