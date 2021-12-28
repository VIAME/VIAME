// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "merge_detection_sets_process.h"

#include <kwiver_type_traits.h>

#include <vital/types/detected_object_set.h>
#include <vital/algo/merge_detections.h>
#include <vital/util/string.h>


namespace kwiver {

create_algorithm_name_config_trait( merger );

// ----------------------------------------------------------------------------
class merge_detection_sets_process::priv
{
public:

  // This is the list of input ports we are reading from.
  std::set< std::string > p_port_list;

  // Possible algorithm for merging
  vital::algo::merge_detections_sptr m_merger;
};

// ----------------------------------------------------------------------------
merge_detection_sets_process
::merge_detection_sets_process( vital::config_block_sptr const& config )
  : process( config )
  , d( new priv )
{
  // This process manages its own inputs.
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}

merge_detection_sets_process
::~merge_detection_sets_process()
{
}

// ----------------------------------------------------------------------------
void merge_detection_sets_process
::_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if( vital::algo::merge_detections::check_nested_algo_configuration_using_trait(
         merger,
         algo_config ) )
  {
    vital::algo::merge_detections::set_nested_algo_configuration_using_trait(
      merger,
      algo_config,
      d->m_merger );
  }
}

// ----------------------------------------------------------------------------
// Post connection processing
void
merge_detection_sets_process
::_init()
{
  // Now that we have a "normal" output port, let Sprokit manage it
  this->set_data_checking_level( check_valid );
}

// ----------------------------------------------------------------------------
void
merge_detection_sets_process
::_step()
{
  // process instrumentation does not make much sense here since most
  // of the time will be spent waiting for input. One approach is to
  // sum up all the time spent adding the input sets to the output
  // sets, but that is not very interesting.

  std::vector< vital::detected_object_set_sptr > inputs;
  vital::detected_object_set_sptr output;

  for ( const auto port_name : d->p_port_list )
  {
    inputs.push_back(
      grab_from_port_as< vital::detected_object_set_sptr >( port_name ) );
  }

  if( d->m_merger )
  {
    output = d->m_merger->merge( inputs );
  }
  else
  {
    output = std::make_shared< vital::detected_object_set >();

    for( auto set : inputs )
    {
      if( set )
      {
        output->add( set );
      }
    }
  }

  push_to_port_using_trait( detected_object_set, output );
}

// ----------------------------------------------------------------------------
void
merge_detection_sets_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, required );
}

// ----------------------------------------------------------------------------
void
merge_detection_sets_process
::make_config()
{
}

// ----------------------------------------------------------------------------
void
merge_detection_sets_process
::input_port_undefined( port_t const& port_name )
{
  LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if( !kwiver::vital::starts_with( port_name, "_" ) )
  {
    // Check for unique port name
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
        port_name,                                 // port name
        detected_object_set_port_trait::type_name, // port type
        required,                                  // port flags
        "detected object set input" );

      d->p_port_list.insert( port_name );
    }
  }
}

} // end namespace
