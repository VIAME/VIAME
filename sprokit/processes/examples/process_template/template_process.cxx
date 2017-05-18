/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
#include "template_process.h"

#include <sprokit/processes/kwiver_type_traits.h>
#include <vital/types/timestamp.h>
#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>

//++ You can put all of your processes in the same namespace
namespace group_ns {

//++ Insert process documentation here. Don't be shy about adding detail.
//++ This description is collected by doxygen for the on-line documentation.
//++ Since the header file is mostly boiler-plate, the process docomentation
//++ is in the implementation file where it is more likely to be read and updated.
// ----------------------------------------------------------------
/**
 * \class template_process
 *
 * \brief Process template
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
create_config_trait( header, std::string, "top", "Header text for image display." );
create_config_trait( footer, std::string, "bottom", "Footer text for image display. Displayed centered at bottom of image." );
create_config_trait( gsd, double, "3.14159", "Meters per pixel scaling." );

//----------------------------------------------------------------
// Private implementation class
class template_process::priv
{
public:
  priv();
  ~priv();

  cv::Mat process_image( cv::Mat img ) { return img; }

  // Configuration values
  std::string m_header;
  std::string m_footer;
  double m_gsd;
}; // end priv class

// ================================================================
//++ This is the standard form for a constructor.
template_process
::template_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new template_process::priv )
{
  attach_logger( kwiver::vital::get_logger( name() ) );
  make_ports(); // create process ports
  make_config(); // declare process configuration
}


template_process
::~template_process()
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
template_process
::_configure()
{
  start_configure_processing();

  //++ Use config traits to access the value for the parameters.
  //++ Values are usually stored in the private structure.
  //++ These config items are not really used in this process.
  //++ There are shown here as an example.
  d->m_header = config_value_using_trait( header );
  d->m_footer = config_value_using_trait( footer );
  d->m_gsd    = config_value_using_trait( gsd ); // converted to double

  stop_configure_processing();
}


// ----------------------------------------------------------------
void
template_process
::_step()
{
  kwiver::vital::timestamp frame_time;

  // See if optional input port has been connected.
  // Get input only if connected.
  //++ Best practice - checking if an optional input port is connected.
  if ( has_input_port_edge_using_trait( timestamp ) )
  {
    frame_time = grab_from_port_using_trait( timestamp );
  }

  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );

  LOG_DEBUG( logger(), "Processing frame " << frame_time );

  // Process Instrumentation call should be just before the real core
  // of the process step processing. It must be after getting the
  // inputs because those calls can stall until inputs are available.
  start_step_processing();

  cv::Mat in_image = kwiver::arrows::ocv::image_container::vital_to_ocv( img->get_image() );

  //++ Here is where the process does its work.
  kwiver::vital::image_container_sptr out_image (new kwiver::arrows::ocv::image_container( d->process_image( in_image ) ) );

  // Process Instrumentation call should be after all step core
  // processing is complete. It must be before pushing the outputs
  // because those calls can stall.
  stop_step_processing();

  push_to_port_using_trait( image, out_image );
}


// ------------------------------------------------------------------
//++ This method is called after all connections have been made to
//++ this process. The process can analyze connections and adapt its
//++ processing if needed. This is not usually needed for basic processes.
//++ If post-connection processing is not needed, delete this method.
void
template_process
::_init()
{
  start_init_processing();

  // do initialization, if applicable.

  stop_init_processing();
}


// ------------------------------------------------------------------
//++ This method is called when the pipeline is reset.
void
template_process
::_reset()
{
  start_reset_processing();

  // do reset processing if applicable.

  stop_reset_processing();
}


// ------------------------------------------------------------------
//++ This method is called when there is a flush on one of the input ports.
//++ Flush usually indicates a break in the data flow; an end of one stream
//++ and start of a new one.
void
template_process
::_flush()
{
  start_flush_processing();

  // perform flush processing if applicable.

  stop_flush_processing();
}


// ------------------------------------------------------------------
//++ This method is called when the pipeline is reconfigured. This is where new
//++ configuration values are supplied to the process. Use reconfig_value_using_trait( conf, ... )
//++ to get the new config values from the supplied config
void
template_process
::_reconfigure(kwiver::vital::config_block_sptr const& conf)
{
  start_reconfigure_processing();

  // perform reconfigure processing if applicable.

  stop_reconfigure_processing();
}



// ----------------------------------------------------------------
void
template_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( image, optional );
}


// ----------------------------------------------------------------
void
template_process
::make_config()
{
  //++ Declaring config items using the traits created at the top of the file.
  //++ This makes the configuration items visible to sprokit pipeline framework.
  declare_config_using_trait( header );
  declare_config_using_trait( footer );
  declare_config_using_trait( gsd );
}


// ================================================================ ++
//Initialize any private data here
template_process::priv
::priv()
  : m_gsd( 0.1122 )
{
}


template_process::priv
::~priv()
{
}

} // end namespace
