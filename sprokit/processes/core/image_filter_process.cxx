// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "image_filter_process.h"

#include <vital/algo/image_filter.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( filter );

//----------------------------------------------------------------
// Private implementation class
class image_filter_process::priv
{
public:
  priv();
  ~priv();

   vital::algo::image_filter_sptr m_filter;

}; // end priv class

// ==================================================================
image_filter_process::
image_filter_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new image_filter_process::priv )
{
  make_ports();
  make_config();
}

image_filter_process::
~image_filter_process()
{
}

// ------------------------------------------------------------------
void
image_filter_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  vital::algo::image_filter::set_nested_algo_configuration_using_trait(
    filter,
    algo_config,
    d->m_filter );

  if ( ! d->m_filter )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create filter" );
  }

  vital::algo::image_filter::get_nested_algo_configuration_using_trait(
    filter,
    algo_config,
    d->m_filter );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::image_filter::check_nested_algo_configuration_using_trait(
         filter, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }
}

// ------------------------------------------------------------------
void
image_filter_process::
_step()
{
  vital::image_container_sptr input = grab_from_port_using_trait( image );

  vital::image_container_sptr result;

  {
    scoped_step_instrumentation();

    // Get detections from filter on image
    result = d->m_filter->filter( input );
  }

  push_to_port_using_trait( image, result );
}

// ------------------------------------------------------------------
void
image_filter_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // We are outputting a shared ref to the output image, therefore we
  // should mark it as shared.
  sprokit::process::port_flags_t output;
  output.insert( flag_output_shared );

  // -- input --
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( image, output );
}

// ------------------------------------------------------------------
void
image_filter_process::
make_config()
{
  declare_config_using_trait( filter );
}

// ================================================================
image_filter_process::priv
::priv()
{
}

image_filter_process::priv
::~priv()
{
}

} // end namespace kwiver
