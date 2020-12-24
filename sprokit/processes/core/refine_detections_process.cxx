// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "refine_detections_process.h"

#include <vital/algo/refine_detections.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( refiner );

//----------------------------------------------------------------
// Private implementation class
class refine_detections_process::priv
{
public:
  priv();
  ~priv();

   vital::algo::refine_detections_sptr m_refiner;

}; // end priv class

// ==================================================================
refine_detections_process::
refine_detections_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new refine_detections_process::priv )
{
  make_ports();
  make_config();
}

refine_detections_process::
~refine_detections_process()
{
}

// ------------------------------------------------------------------
void
refine_detections_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::refine_detections::check_nested_algo_configuration_using_trait(
         refiner, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  vital::algo::refine_detections::set_nested_algo_configuration_using_trait(
    refiner,
    algo_config,
    d->m_refiner );

  if ( ! d->m_refiner )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create refiner" );
  }
}

// ------------------------------------------------------------------
void
refine_detections_process::
_step()
{
  vital::image_container_sptr image = grab_from_port_using_trait( image );
  vital::detected_object_set_sptr dets = grab_from_port_using_trait( detected_object_set );

  vital::detected_object_set_sptr results;
  {
    scoped_step_instrumentation();

    // Get detections from refiner on image
    results = d->m_refiner->refine( image, dets );
  }

  push_to_port_using_trait( detected_object_set, results );
}

// ------------------------------------------------------------------
void
refine_detections_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
}

// ------------------------------------------------------------------
void
refine_detections_process::
make_config()
{
  declare_config_using_trait( refiner );
}

// ================================================================
refine_detections_process::priv
::priv()
{
}

refine_detections_process::priv
::~priv()
{
}

} // end namespace kwiver
