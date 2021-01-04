// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
#include <vital/vital_config.h>
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
create_config_trait( header, std::string, "top", "Header text for image display." );
create_config_trait( footer, std::string, "bottom",
                     "Footer text for image display. Displayed centered at bottom of image." );
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
  scoped_configure_instrumentation();

  //++ Use config traits to access the value for the parameters.
  //++ Values are usually stored in the private structure.
  //++ These config items are not really used in this process.
  //++ There are shown here as an example.
  d->m_header = config_value_using_trait( header );
  d->m_footer = config_value_using_trait( footer );
  d->m_gsd    = config_value_using_trait( gsd ); // converted to double
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

  kwiver::vital::image_container_sptr out_image;

  // Process Instrumentation call should be just before the real core
  // of the process step processing. It must be after getting the
  // inputs because those calls can stall until inputs are available.
  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Processing frame " << frame_time );
    using namespace kwiver::arrows::ocv;

    cv::Mat in_image = image_container::vital_to_ocv( img->get_image(),
                                                      image_container::RGB_COLOR );

    //++ Here is where the process does its work.
    out_image = std::make_shared<image_container>( d->process_image( in_image ),
                                                   image_container::RGB_COLOR );
  }

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
  scoped_init_instrumentation();

  // do initialization, if applicable.

}

// ------------------------------------------------------------------
//++ This method is called when the pipeline is reset.
void
template_process
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
template_process
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
template_process
::_reconfigure( VITAL_UNUSED kwiver::vital::config_block_sptr const& conf)
{
  scoped_reconfigure_instrumentation();

  // perform reconfigure processing if applicable.

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

  sprokit::process::port_flags_t shared;
  shared.insert( flag_output_shared );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, required );

  // -- output --
  // since images are passed by shared ptr, they must be
  // marked as shared.
  declare_output_port_using_trait( image, shared );
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
