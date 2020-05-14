/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
//++ The above header applies to this template code. Feel free to use your own
//++ license header.

/**
 * \file
 * \brief Implementation of template process.
 */

//++ include process class/interface
#include "template_algo_wrapper.h"

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>
#include <vital/types/timestamp.h>
#include <arrows/ocv/image_container.h>

//++ include definition of abstract base algorithm.
#include <vital/algo/image_object_detector.h>


//++ You can put all of your processes in the same namespace
namespace group_ns {

//++ Insert process documentation here. Don't be shy about adding detail.
//++ This description is collected by doxygen for the on-line documentation.
//++ Since the header file is mostly boiler-plate, the process docomentation
//++ is in the implementation file where it is more likely to be read and updated.
// ----------------------------------------------------------------
/**
 * \class template_algo_wrapper
 *
 * \brief Process template
 *
 * \process Full description of process.
 *
 * \iports
 *
 * \iport{timestamp} time stamp for incoming images.
 *
 * \iport{image} Input image to be processed.
 *
 * \oports
 *
 * \oport{image} Resulting image
 *
 * \configs
 *
 * \config{header} Header text. (string)
 *
 * \config{footer} Footer text (string)
 */

//++ Configuration traits are used to declare configurable items. Be sure to supply a complete
//++ description/specification of the item, since this will serve as the primary documentation
//++ for the parameter. Feel free to use multiple lines, but no new-lines are needed since this
//++ text is wrapped when it is displayed.
//++ Define configurable items here.
// config items
// <name>, <type>, <default string>, <description string>
create_config_trait( algo_name, std::string, "", "Name of algorithm config block." );

//----------------------------------------------------------------
// Private implementation class
class template_algo_wrapper::priv
{
public:
  priv();
  ~priv();

  //++ This is a pointer to the abstract base class of the algorithm
  //++ type that this process wraps. The base class pointer is needed
  //++ here because the actual derived class is determined from the run
  //++ time config entries.
  kwiver::vital::algo::image_object_detector_sptr m_algo;

}; // end priv class

// ================================================================
//++ This is the standard form for a constructor.
template_algo_wrapper
::template_algo_wrapper( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new template_algo_wrapper::priv )
{
  make_ports(); // create process ports
  make_config(); // declare process configuration
}


template_algo_wrapper
::~template_algo_wrapper()
{
}


// ----------------------------------------------------------------
/**
 * @brief Configure process
 *
 * This method is called prior to connecting ports to allow the
 * process to configure itself.
 */
void
template_algo_wrapper
::_configure()
{
  scoped_configure_instrumentation();

  // Get process configurartion block.
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  //++ The name supplied here must match the one defined in the coinfig_trait defined above.
  //++ Note that these methods are static on the abstract base algorithm type.
  if ( ! kwiver::vital::algo::image_object_detector::check_nested_algo_configuration( "algo_name", algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  kwiver::vital::algo::image_object_detector::set_nested_algo_configuration( "algo_name", algo_config, d->m_algo );
  if ( ! d->m_algo )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create algorithm." );
  }
}


// ----------------------------------------------------------------
void
template_algo_wrapper
::_step()
{
  kwiver::vital::timestamp frame_time;

  // See if optional input port has been connected.
  // Get input only if connected.
  //++ Best practice - checking if an optional input port is connected.
  //++ This example does not actually use the timestamp, but it is here
  //++ to show how optional inputs are handled.
  if ( has_input_port_edge_using_trait( timestamp ) )
  {
    frame_time = grab_from_port_using_trait( timestamp );
  }

  kwiver::vital::image_container_sptr in_image = grab_from_port_using_trait( image );

  kwiver::vital::detected_object_set_sptr out_set;

  // Process Instrumentation call should be just before the real core
  // of the process step processing. It must be after getting the
  // inputs because those calls can stall until inputs are available.
  {
    scoped_step_instrumentation();

    //++ Here is where the process does its work.
    out_set = d->m_algo->detect( in_image );
  }

  push_to_port_using_trait( detected_object_set, out_set );
}


// ------------------------------------------------------------------
//++ This method is called after all connections have been made to
//++ this process. The process can analyze connections and adapt its
//++ processing if needed. This is not usually needed for basic processes.
//++ If post-connection processing is not needed, delete this method.
void
template_algo_wrapper
::_init()
{
  scoped_init_instrumentation();

  // do initialization, if applicable.

}


// ------------------------------------------------------------------
//++ This method is called when the pipeline is reset.
void
template_algo_wrapper
::_reset()
{
  scoped_reset_instrumentation();

  // do reset processing if applicable.

}


// ------------------------------------------------------------------
//++ This method is called when there is a flush on one of the input ports.
//++ Flush usually indicates a break in the data flow; an end of one stream
//++ and start of a new one.
void
template_algo_wrapper
::_flush()
{
  scoped_flush_instrumentation();

  // perform flush processing if applicable.

}


// ------------------------------------------------------------------
//++ This method is called when the pipeline is reconfigured. This is where new
//++ configuration values are supplied to the process. Use reconfig_value_using_trait( conf, ... )
//++ to get the new config values from the supplied config
void
template_algo_wrapper
::_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  scoped_reconfigure_instrumentation();

  // perform reconfigure processing if applicable.

}


// ----------------------------------------------------------------
void
template_algo_wrapper
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
}


// ----------------------------------------------------------------
void
template_algo_wrapper
::make_config()
{
  //++ Declaring config items using the traits created at the top of the file.
  //++ This makes the configuration items visible to sprokit pipeline framework.
  declare_config_using_trait( algo_name );
}


// ================================================================ ++
//Initialize any private data here
template_algo_wrapper::priv
::priv()
{
}


template_algo_wrapper::priv
::~priv()
{
}

} // end namespace
