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

/**
 * \file
 * \brief Image display process implementation.
 */

#include "image_viewer_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sstream>
#include <iostream>


namespace kwiver {
// ----------------------------------------------------------------
/**
 * \class view_image_process
 *
 * \brief Display image on screen.
 *
 * This process displays the input image with optional
 * annotations. The best use for this process is for monitoring images
 * in the pipeline.
 *
 * \iports
 *
 * \iport{timestamp} Timestamp of image. This data is used to annotate image.
 *
 * \iport{image} Image to display.
 *
 * \configs
 *
 * \config{pause_time} Seconds to pause between displaying images. A
 * value of 0 waits for keyboard input.
 *
 * \config{annotate_image} Boolean indicating if image is to be annotated. If \b true
 * then the frame number, header and footer are added to the displayed image.
 *
 * \config{header} Text to be written in the top boarder of the display.
 *
 * \config{footer} Text to be written in the bottom boarder of the display.
 */

// config items
  // <name>, <type>, <default string>, <description>
create_config_trait( pause_time, float, "0", "Interval to pause between frames. 0 means wait for keystroke, "
                     "Otherwise interval is in seconds (float)" );
create_config_trait( annotate_image, bool, "false", "Add frame number and other text to display." );
create_config_trait( title, std::string, "Display window", "Display window title text.." );
create_config_trait( header, std::string, "", "Header text for image display." );
create_config_trait( footer, std::string, "", "Footer text for image display. Displayed centered at bottom of image." );

//----------------------------------------------------------------
// Private implementation class
class image_viewer_process::priv
{
public:
  priv();
  ~priv();


  // Configuration values
  int m_pause_ms;
  bool m_annotate_image;
  std::string m_title;
  std::string m_header;
  std::string m_footer;


  // ------------------------------------------------------------------
  cv::Mat
  annotate_image( cv::Mat cv_img, kwiver::vital::timestamp::frame_t frame)
  {
    static const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    static const double font_scale( 1.0 );
    static const int font_thickness( 2 );

    std::stringstream display_text;

    display_text << "Frame: " << frame;

    // Get text box size
    int baseline( 0 );
    cv::Size tbox = cv::getTextSize( display_text.str(), // text
                                     font_face, // font code number
                                     font_scale, // Font scale factor that is multiplied by the font-specific base size.
                                     font_thickness, // Thickness of lines used to render the text.
                                     &baseline ); // o: y-coordinate of the baseline relative to the bottom-most text point.
    cv::Mat image;

    // Add borders to the image top and bottom
    cv::copyMakeBorder( cv_img, image, // input, output images
                        tbox.height + 8, tbox.height + 8,     // top, bottom
                        0, 0,     // left, right
                        cv::BORDER_CONSTANT,
                        cv::Scalar::all( 255 ) ); // White fill

    // Put this in the top border
    cv::Point text_org( 5, tbox.height + 3 );

    cv::putText( image,             // image array
                 display_text.str(), // text to display
                 text_org,          // bottom left corner of text
                 font_face,         // font face
                 font_scale,        // font scale
                 cv::Scalar::all( 10 ), // text color
                 font_thickness );  // Thickness of the lines used to draw a text

    // header
    if ( ! m_header.empty() )
    {
      cv::Size tbox = cv::getTextSize( m_header,
                                       font_face,
                                       font_scale,
                                       font_thickness,
                                       &baseline );

      // Calculate point for lower left of text block
      cv::Point header_org( ( image.cols - tbox.width ) / 2,
                            tbox.height + 3 );

      cv::putText( image,
                   m_header,
                   header_org,
                   font_face,
                   font_scale,
                   cv::Scalar::all( 10 ),
                   font_thickness );
    }

    // footer
    if ( ! m_footer.empty() )
    {
      cv::Size tbox = cv::getTextSize( m_footer,
                                       font_face,
                                       font_scale,
                                       font_thickness,
                                       &baseline );

      // Calculate point for lower left of text block
      cv::Point footer_org( ( image.cols - tbox.width ) / 2,
                            ( image.rows - 3 ) );

      cv::putText( image,
                   m_footer,
                   footer_org,
                   font_face,
                   font_scale,
                   cv::Scalar::all( 10 ),
                   font_thickness );
    }

    return image;
  } // annotate_image

}; // end priv class


// ================================================================

image_viewer_process
::image_viewer_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new image_viewer_process::priv )
{
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach
  make_ports();
  make_config();
}


image_viewer_process
::~image_viewer_process()
{
}


// ----------------------------------------------------------------
void
image_viewer_process
::_configure()
{
  d->m_pause_ms = static_cast< int >( config_value_using_trait( pause_time ) * 1000.0 ); // convert to msec
  d->m_annotate_image = config_value_using_trait( annotate_image );
  d->m_title          = config_value_using_trait( title );
  d->m_header         = config_value_using_trait( header );
  d->m_footer         = config_value_using_trait( footer );
}


// ----------------------------------------------------------------
void
image_viewer_process
::_step()
{
  kwiver::vital::timestamp frame_time;

  // Test to see if optional port is connected.
  if (has_input_port_edge_using_trait( timestamp ) )
  {
    frame_time = grab_input_using_trait( timestamp );
  }

  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );

  LOG_DEBUG( logger(), "Processing frame " << frame_time );

  cv::Mat image = arrows::ocv::image_container::vital_to_ocv( img->get_image(), arrows::ocv::image_container::BGR );

  if ( d->m_annotate_image )
  {
    image = d->annotate_image( image, frame_time.get_frame() );
  }

  cv::namedWindow( d->m_title, cv::WINDOW_NORMAL ); // Create a window for display.
  cv::imshow( d->m_title, image ); // Show our image inside it.

  cv::waitKey( d->m_pause_ms );
}


// ----------------------------------------------------------------
void
image_viewer_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, required );
}


// ----------------------------------------------------------------
void
image_viewer_process
::make_config()
{
  declare_config_using_trait( pause_time );
  declare_config_using_trait( annotate_image );
  declare_config_using_trait( title );
  declare_config_using_trait( header );
  declare_config_using_trait( footer );
}


// ================================================================
image_viewer_process::priv
::priv()
  : m_pause_ms( 0 ),
    m_annotate_image( false )
{
}


image_viewer_process::priv
::~priv()
{
}

} // end namespace
