// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of the detected object set filter process.
 */

#include "detected_object_filter_process.h"

#include <vital/algo/detected_object_filter.h>
#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include <sstream>
#include <iostream>

namespace kwiver {

create_algorithm_name_config_trait( filter );

// ----------------------------------------------------------------
/**
 * \class detected_object_filter_process
 *
 * \brief Filter detected image object sets.
 *
 * \process This process filters a set of detected image objects and
 * produces a new set of detected image objects. The actual processing
 * is done by the selected \b detected_object_filter algorithm
 * implementation.
 *
 * \iports
 *
 * \iport{detected_object_set} Set of objects to be passed to the
 * filtering algorithm.
 *
 * \oports
 *
 * \oport{detected_object_set} SEt of objects produced by the
 * filtering algorithm.
 *
 * \configs
 *
 * \config{filter} Name of the configuration subblock that selects
 * and configures the drawing algorithm.
 */

//----------------------------------------------------------------
// Private implementation class
class detected_object_filter_process::priv
{
public:
  priv();
  ~priv();

  vital::algo::detected_object_filter_sptr m_filter;

}; // end priv class

// ================================================================

detected_object_filter_process
::detected_object_filter_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
  , d( new detected_object_filter_process::priv )
{
  make_ports();
  make_config();
}

detected_object_filter_process
::~detected_object_filter_process()
{
}

// ----------------------------------------------------------------
void
detected_object_filter_process
::_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::detected_object_filter::check_nested_algo_configuration_using_trait(
         filter, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  vital::algo::detected_object_filter::set_nested_algo_configuration_using_trait(
    filter,
    algo_config,
    d->m_filter );

  if ( ! d->m_filter )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create filter" );
  }
}

// ----------------------------------------------------------------
void
detected_object_filter_process
::_step()
{
  vital::detected_object_set_sptr input = grab_from_port_using_trait( detected_object_set );

  vital::detected_object_set_sptr result;

  {
    scoped_step_instrumentation();

    result = d->m_filter->filter( input );
  }

  push_to_port_using_trait( detected_object_set, result );
}

// ----------------------------------------------------------------
void
detected_object_filter_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
}

// ----------------------------------------------------------------
void
detected_object_filter_process
::make_config()
{
  declare_config_using_trait( filter );
}

// ================================================================
detected_object_filter_process::priv
::priv()
{
}

detected_object_filter_process::priv
::~priv()
{
}

} //end namespace
