/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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

#include "supply_image.h"
#include "io_mgr.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/algorithm_plugin_manager.h>

#include <arrows/algorithms/ocv/image_container.h>

#include <arrows/processes/kwiver_type_traits.h>

// -- DEBUG
#if defined DEBUG
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif

namespace kwiver {

//----------------------------------------------------------------
// Private implementation class
class supply_image::priv
{
public:
  priv();
  ~priv();

  bool first;
}; // end priv class


// ================================================================

supply_image
::supply_image( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new supply_image::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach
  kwiver::vital::algorithm_plugin_manager::load_plugins_once();

  make_ports();
  make_config();
}


supply_image
::~supply_image()
{
}


// ----------------------------------------------------------------
void supply_image
::_configure()
{
}


// ----------------------------------------------------------------
void supply_image
::_step()
{
  LOG_DEBUG( logger(), "supplying image" );

  if ( d->first )
  {
    d->first = false;

    // Convert image to a image container.
    kwiver::vital::image_container_sptr const img_c( new kwiver::arrows::ocv::image_container( io_mgr::Instance()->GetImage() ) );

    // --- debug
#if defined DEBUG
    cv::Mat image = maptk::ocv::image_container::maptk_to_ocv( img_c->get_image() );
    namedWindow( "Display window", cv::WINDOW_NORMAL );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                 // Wait for a keystroke in the window
#endif
    // -- end debug

    push_to_port_using_trait( image, img_c );
  }
  else
  {
    LOG_DEBUG( logger(), "End of input reached, process terminating" );
    // indicate done
    mark_process_as_complete();
    const sprokit::datum_t dat= sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( image, dat );
  }
}


// ----------------------------------------------------------------
void supply_image
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  declare_output_port_using_trait( image, required );
}


// ----------------------------------------------------------------
void supply_image
::make_config()
{
}


// ================================================================
supply_image::priv
::priv()
  : first(true)
{
}


supply_image::priv
::~priv()
{
}


} // end namespace
