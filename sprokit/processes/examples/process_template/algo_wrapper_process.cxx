// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "algo_wrapper_process.h"

#include <vital/algo/refine_detections.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

/*
 * Note that the //+ comments are intended for the person who is
 * adapting this template and should be removed from the final
 * product.
 */

namespace kwiver {

//+ Define a config entry for the main algorithm configuration
//+ subblock. A name other than "algorithm" can be used if it is more
//+ descriptive. If so, change name throughout the rest of this file also.
  create_algorithm_name_config_trait( algo );

//----------------------------------------------------------------
// Private implementation class
class algo_wrapper_process::priv
{
public:
  priv();
  ~priv();

  //+ define sptr to specific algorithm type
  vital::algo::algo_type_sptr m_algo;

}; // end priv class

// ==================================================================
algo_wrapper_process::
algo_wrapper_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new algo_wrapper_process::priv )
{
  make_ports();
  make_config();
}

algo_wrapper_process::
~algo_wrapper_process()
{
}

// ------------------------------------------------------------------
void
algo_wrapper_process::
_configure()
{
  start_configure_processing();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems.
  // Call check_nested_algo_configuration() first so that it will display a list of
  // concrete instances of the desired algorithms that are available if the config
  // does not select a valid one.
  if ( ! vital::algo::refine_detections::check_nested_algo_configuration_using_trait(
         algo, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  vital::algo::refine_detections::set_nested_algo_configuration_using_trait(
    algo, algo_config, d->m_algo );

  if ( ! d->m_algo )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create algorithm" );
  }

  stop_configure_processing();
}

// ------------------------------------------------------------------
void
algo_wrapper_process::
_step()
{
  //+ get inputs as needed
  vital::image_container_sptr image = grab_from_port_using_trait( image );
  vital::detected_object_set_sptr dets = grab_from_port_using_trait( detected_object_set );

  start_step_processing();      // Mark start of productive processing

  // Send inputs to algorithm

  //+ This call must correspond to the wrapped algorithm.
  vital::detected_object_set_sptr results = d->m_algo->process( image, dets );

  stop_step_processing();       // Mark end of productive processing

  //+ push output as needed
  push_to_port_using_trait( detected_object_set, results );
}

// ------------------------------------------------------------------
void
algo_wrapper_process::
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
algo_wrapper_process::
make_config()
{
  declare_config_using_trait( slgorithm );
}

// ================================================================
algo_wrapper_process::priv
::priv()
{
}

algo_wrapper_process::priv
::~priv()
{
}

} // end namespace kwiver
