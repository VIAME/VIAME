// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "split_image_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>

#include <vital/algo/split_image.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( split_image );

//----------------------------------------------------------------
// Private implementation class
class split_image_process::priv
{
public:
  priv();
  ~priv();

  algo::split_image_sptr         m_image_splitter;
};

// ================================================================

split_image_process
::split_image_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new split_image_process::priv )
{
  make_ports();
  make_config();
}

split_image_process
::~split_image_process()
{
}

// ----------------------------------------------------------------
void split_image_process
::_configure()
{
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::split_image::set_nested_algo_configuration_using_trait(
    split_image, algo_config, d->m_image_splitter );

  if( !d->m_image_splitter )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
      name(), "Unable to create \"split_image\"" );
  }
  algo::split_image::get_nested_algo_configuration_using_trait(
    split_image, algo_config, d->m_image_splitter );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::split_image::check_nested_algo_configuration_using_trait(
        split_image, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
      "Configuration check failed." );
  }
}

// ----------------------------------------------------------------
void
split_image_process
::_step()
{
  // Get image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  std::vector< kwiver::vital::image_container_sptr > outputs;

  // Get feature tracks
  outputs = d->m_image_splitter->split( img );

  // Return by value
  if( outputs.size() >= 1 )
  {
    push_to_port_using_trait( left_image, outputs[0] );
  }

  if( outputs.size() >= 2 )
  {
    push_to_port_using_trait( right_image, outputs[1] );
  }
}

// ----------------------------------------------------------------
void split_image_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );

  declare_output_port_using_trait( left_image, optional );
  declare_output_port_using_trait( right_image, optional );
}

// ----------------------------------------------------------------
void split_image_process
::make_config()
{
  declare_config_using_trait( split_image );
}

// ================================================================
split_image_process::priv
::priv()
{
}

split_image_process::priv
::~priv()
{
}

} // end namespace
