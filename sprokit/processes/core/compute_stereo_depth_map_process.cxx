// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "compute_stereo_depth_map_process.h"

#include <vital/algo/compute_stereo_depth_map.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( computer );

// ----------------------------------------------------------------
/**
 * \class compute_stereo_depth_map_process
 *
 * \brief Compute stereo depth map image.
 *
 * \process This process generates a depth map image from a pair of
 * stereo images. The actual calculation is done by the selected \b
 * compute_stereo_depth_map algorithm implementation.
 *
 * \iports
 *
 * \iport{left_image} Left image of the stereo image pair.
 *
 * \iport{right_image} Right image if the stereo pair.
 *
 * \oports
 *
 * \oport{depth_map} Resulting depth map.
 *
 * \configs
 *
 * \config{computer} Name of the configuration subblock that selects
 * and configures the algorithm.
 */

//----------------------------------------------------------------
// Private implementation class
class compute_stereo_depth_map_process::priv
{
public:
  priv();
  ~priv();

   vital::algo::compute_stereo_depth_map_sptr m_computer;

}; // end priv class

// ==================================================================
compute_stereo_depth_map_process::
compute_stereo_depth_map_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new compute_stereo_depth_map_process::priv )
{
  make_ports();
  make_config();
}

compute_stereo_depth_map_process::
~compute_stereo_depth_map_process()
{
}

// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  vital::algo::compute_stereo_depth_map::set_nested_algo_configuration_using_trait(
    computer,
    algo_config,
    d->m_computer );

  if ( ! d->m_computer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create computer" );
  }

  vital::algo::compute_stereo_depth_map::get_nested_algo_configuration_using_trait(
    computer,
    algo_config,
    d->m_computer );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::compute_stereo_depth_map::check_nested_algo_configuration_using_trait(
         computer, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }
}

// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
_step()
{
  vital::image_container_sptr left_image =
      grab_from_port_using_trait( left_image );
  vital::image_container_sptr right_image =
      grab_from_port_using_trait( right_image );

  vital::image_container_sptr depth_map;

  {
    scoped_step_instrumentation();

    // Get detections from computer on image
    depth_map = d->m_computer->compute( left_image, right_image );
  }

  push_to_port_using_trait( depth_map, depth_map );
}

// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( left_image, required );
  declare_input_port_using_trait( right_image, required );

  // -- output --
  declare_output_port_using_trait( depth_map, optional );
}

// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
make_config()
{
  declare_config_using_trait( computer );
}

// ================================================================
compute_stereo_depth_map_process::priv
::priv()
{
}

compute_stereo_depth_map_process::priv
::~priv()
{
}

} // end namespace kwiver
