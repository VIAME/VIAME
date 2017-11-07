/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * \brief Implementation for draw_detected_object_set
 */

#include "draw_detected_object_set.h"

#include <vital/vital_types.h>
#include <vital/util/tokenize.h>

#include <kwiversys/RegularExpression.hxx>
#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace ocv {

// Constant for offsetting drawn labels
static const int MULTI_LABEL_OFFSET(15);

typedef  Eigen::Matrix< unsigned int, 3, 1 > ColorVector;

// ==================================================================
/**
 * @brief
 *
 */
class draw_detected_object_set::priv
{
public:
  // -- CONSTRUCTORS --
  priv()
    : m_config_error( false )
    , m_threshold( -1 )
    , m_do_alpha( true )
    , m_text_scale( 0.4 )
    , m_text_thickness( 1.0 )
    , m_clip_box_to_image( false )
    , m_draw_text( true )
  {
    m_default_params.thickness = 1.0;
    m_default_params.color[0] = 255;
    m_default_params.color[1] = 0;
    m_default_params.color[2] = 0;
  }

  ~priv()
  { }

  // internal state
  bool m_config_error;

  // Configuration values
  float m_threshold;
  std::vector< std::string > m_select_classes;
  bool m_do_alpha;

  struct Bound_Box_Params
  {
    float thickness;
    ColorVector color;
  } m_default_params;

  // box attributes per object type
  std::map< std::string, Bound_Box_Params > m_custum_colors;

  float m_text_scale;
  float m_text_thickness;
  bool m_clip_box_to_image;
  bool m_draw_text;

  // -- temp config storage --
  std::string m_tmp_custom;
  std::string m_tmp_def_color;
  std::string m_tmp_class_select;

  draw_detected_object_set* m_parent;



  // ------------------------------------------------------------------
  /**
   * @brief Draw a box on an image.
   *
   * This method draws a box on an image for the bounding box from a
   * detected object.
   *
   * When drawing a box with multiple class names, draw the first
   * class_name with the \c just_text parameter \b false and all
   * subsequent calls with it set to \b true. Also the \c offset
   * parameter must be incremented so the labels do not overwrite.
   *
   * @param[in,out] image Input image updated with drawn box
   * @param[in] dos detected object with bounding box
   * @param[in] label Text label to use for box
   * @param[in] prob Probability value to add to label text
   * @param[in] just_text Set to true if only draw text, not the
   *            bounding box. This is used when there are multiple
   *            labels for the same detection.
   * @param[in] offset How much to offset text fill box from text
   *            baseline. This is used to offset labels when there are
   *            more than one label for a detection.
   */
  void draw_box( cv::Mat&                     image,
                 const vital::detected_object_sptr  dos,
                 std::string                  label,
                 double                       prob,
                 bool                         just_text = false,
                 int                          offset_index = 0 ) const
  {
    cv::Mat overlay;

    image.copyTo( overlay );
    vital::bounding_box_d bbox = dos->bounding_box();
    if ( m_clip_box_to_image )
    {
      cv::Size s = image.size();
      vital::bounding_box_d img( vital::bounding_box_d::vector_type( 0, 0 ),
                                 vital::bounding_box_d::vector_type( s.width, s.height ) );
      bbox = intersection( img, bbox );
    }

    // Make CV rect for out bbox coordinates
    cv::Rect r( bbox.upper_left()[0], bbox.upper_left()[1], bbox.width(), bbox.height() );
    std::string p = std::to_string( static_cast<long double>( prob ) ); // convert value to string
    std::string txt = label + " " + p;

    // Clip threshold to limit value. If less than 0.05, leave threshold as it is.
    // Else lower by 5%. This is a heuristic for making the alpha shading look good.
    double tmp_thresh = ( this->m_threshold - ( ( this->m_threshold >= 0.05 ) ? 0.05 : 0 ) );

    double alpha_wight =  ( m_do_alpha ) ? ( ( prob - tmp_thresh ) / ( 1 - tmp_thresh ) ) : 1.0;

    Bound_Box_Params const* bbp = &m_default_params;
    auto iter = m_custum_colors.find( label );

    // look for custom color for this class_name
    if ( iter != m_custum_colors.end() )
    {
      bbp = &( iter->second );
    }

    // Add text to an existing box
    if ( ! just_text )
    {
      cv::Scalar color( bbp->color[0], bbp->color[1], bbp->color[2] );
      cv::rectangle( overlay, r, color, bbp->thickness );
    }

    if ( m_draw_text )
    {
      int fontface = cv::FONT_HERSHEY_SIMPLEX;
      double scale = m_text_scale;
      int thickness = m_text_thickness;
      int baseline = 0;
      cv::Point pt( r.tl() + cv::Point( 0, MULTI_LABEL_OFFSET * offset_index ) );

      cv::Size text = cv::getTextSize( txt, fontface, scale, thickness, &baseline );
      cv::rectangle( overlay, pt + cv::Point( 0, baseline ), pt +
                     cv::Point( text.width, -text.height ), cv::Scalar( 0, 0, 0 ), CV_FILLED );

      cv::putText( overlay, txt, pt, fontface, scale, cv::Scalar( 255, 255, 255 ), thickness, 8 );
    }

    cv::addWeighted( overlay, alpha_wight, image, 1 - alpha_wight, 0, image );
  } // draw_box


  // ------------------------------------------------------------------
  /**
   * @brief Draw detected object on image.
   *
   * This method draws the detections on a copy of the supplied
   * image. The detections are drawn in confidence order up to the
   * threshold. For each detection, the most likely class_name is
   * optionally displayed below the box.
   *
   * @param image_data The image to draw on.
   * @param input_set List of detections to draw.
   *
   * @return New image with boxes drawn.
   */
  vital::image_container_sptr draw_detections( vital::image_container_sptr      image_data,
                                               vital::detected_object_set_sptr  in_set ) const
  {
    cv::Mat image = image_container_to_ocv_matrix( *image_data, arrows::ocv::image_container::BGR ).clone();

    // process the detection set
    auto ie =  in_set->cend();
    for ( auto det = in_set->cbegin(); det != ie; ++det )
    {
      auto det_type = (*det)->type();
      if ( ! det_type )
      {
        // No type has been assigned. Just filter on threshold
        if ((*det)->confidence() < m_threshold )
        {
          continue;
        }

        draw_box( image, *det, "", (*det)->confidence() );
        continue;
      }

      // -----------------------------
      // Since there is a type assigned, select on specified class_names
      auto names = det_type->class_names(); // get all class_names

      bool text_only( false );
      int count( 0 );

      // Draw once for each selected class_name
      for( auto n : names )
      {
        double score = det_type->score( n );
        if ( score < m_threshold || ! name_selected( n ) )
        {
          continue;
        }

        LOG_TRACE( m_parent->logger(), "Drawing box for class: " << n << "   score: " << score );
        draw_box( image, *det, n, score, text_only, count );
        text_only = true; // skip box on all subsequent calls
      }
    } // end foreach

    return vital::image_container_sptr( new arrows::ocv::image_container( image, arrows::ocv::image_container::BGR ) );
  } // end draw_detections


// ------------------------------------------------------------------
  /**
   * @brief See if name has been selected for display.
   *
   * @param name Name to check.
   *
   * @return \b true if name should be rendered
   */
  bool name_selected( std::string const& name ) const
  {
    if ( m_select_classes[0] == "*ALL*" )
    {
      return true;
    }

    return (std::find( m_select_classes.begin(), m_select_classes.end(), name ) != m_select_classes.end() );
  }


// ------------------------------------------------------------------
void
process_config()
{
  // Parse custom class color specification
  // class/line-thickness/color-rgb;class/line-thickness/color-rgb
  // e.g. person/3.5/0 0 255;
  {
    std::vector< std::string > cspec;
    kwiver::vital::tokenize( m_tmp_custom, cspec, ";", true );

    for( auto cs : cspec )
    {
      kwiversys::RegularExpression exp( "\\$([^/]+)/([0-9.]+)/([0-9]+) ([0-9]+) ([0-9]+)" );

      if ( ! exp.find( cs ) )
      {
        // parse error - log something
        m_config_error = true;
        LOG_ERROR( m_parent->logger(), "Error parsing custom color specification \"" << cs << "\"" );

        return;
      }

      // exp.match(0) - whole match
      // exp.match(1) - class_name string
      // exp.match(2) - line thickness
      // exp.match(3) - color red
      // exp.match(4) - color green
      // exp.match(5) - color blue

      draw_detected_object_set::priv::Bound_Box_Params bp;

      bp.thickness = std::stof( exp.match(2) );
      bp.color[0] = std::stoi( exp.match(5) );
      bp.color[1] = std::stoi( exp.match(4) );
      bp.color[2] = std::stoi( exp.match(3) );

      m_custum_colors[exp.match(1)] = bp; // add to map
    } // end foreach
  } // end local scope


  { // parse defaults default color
    kwiversys::RegularExpression exp( "([0-9]+) ([0-9]+) ([0-9]+)" );

      if ( ! exp.find( m_tmp_def_color ) )
      {
        // parse error - log something
        m_config_error = true;
        LOG_ERROR( m_parent->logger(), "Error parsing custom color specification \""
                   << m_tmp_def_color << "\"" );
        return;
      }

      // exp.match(0) - whole match
      // exp.match(1) - color red
      // exp.match(2) - color green
      // exp.match(3) - color blue

      m_default_params.color[0] = std::stoi( exp.match(3) );
      m_default_params.color[1] = std::stoi( exp.match(2) );
      m_default_params.color[2] = std::stoi( exp.match(1) );
  } // end local scope

  // Parse selected class_names
  kwiver::vital::tokenize( m_tmp_class_select, m_select_classes, ";", true );
}

}; // end priv class


// ==================================================================
draw_detected_object_set::
draw_detected_object_set()
  : d( new priv )
{
  d->m_parent = this;
}


draw_detected_object_set::
~draw_detected_object_set()
{ }


// ------------------------------------------------------------------
vital::config_block_sptr
draw_detected_object_set::
get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "threshold", d->m_threshold, "min threshold for output (float). "
                     "Detections with confidence values below this value are not drawn." );
  config->set_value( "alpha_blend_prob", d->m_do_alpha,
                     "If true, those who are less likely will be more transparent." );
  config->set_value( "default_line_thickness", d->m_default_params.thickness,
                     "The default line thickness for a class, in pixels." );
  config->set_value( "default_color", "0 0 255",
                     "The default color for a class (RGB)." );
  config->set_value( "custom_class_color", "",
                     "List of class/thickness/color seperated by semicolon. "
                     "For example: person/3/255 0 0;car/2/0 255 0. "
                     "Color is in RGB.");

  config->set_value( "select_classes", "*ALL*",
                     "List of classes to display, separated by a semicolon. For example: person;car;clam" );
  config->set_value( "text_scale", d->m_text_scale, "Scaling for the text label. "
                     "Font scale factor that is multiplied by the font-specific base size." );
  config->set_value( "text_thickness", d->m_text_thickness,
                     "Thickness of the lines used to draw a text." );

  config->set_value( "clip_box_to_image", d->m_clip_box_to_image,
                     "If this option is set to true, the bounding box is clipped to the image bounds." );
  config->set_value( "draw_text", d->m_draw_text,
                     "If this option is set to true, the class name is drawn next to the detection." );
  return config;
}


// ------------------------------------------------------------------
void
draw_detected_object_set::
set_configuration(vital::config_block_sptr config_in)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  d->m_do_alpha                 = config->get_value< bool >( "alpha_blend_prob" );
  d->m_clip_box_to_image        = config->get_value< bool >( "clip_box_to_image" );
  d->m_tmp_custom               = config->get_value< std::string >( "custom_class_color" );
  d->m_tmp_def_color            = config->get_value< std::string >( "default_color" );
  d->m_default_params.thickness = config->get_value< float >( "default_line_thickness" );
  d->m_draw_text                = config->get_value< bool >( "draw_text" );
  d->m_tmp_class_select         = config->get_value< std::string >( "select_classes" );
  d->m_text_scale               = config->get_value< float >( "text_scale" );
  d->m_text_thickness           = config->get_value< float >( "text_thickness" );
  d->m_threshold                = config->get_value< float >( "threshold" );

  d->process_config();
}


// ------------------------------------------------------------------
bool
draw_detected_object_set::
check_configuration(vital::config_block_sptr config) const
{
  // This can be called before the config is "set". A more robust way
  // of determining validity should be used.
  return ! d->m_config_error;
}


// ------------------------------------------------------------------
kwiver::vital::image_container_sptr
draw_detected_object_set::
draw( kwiver::vital::detected_object_set_sptr detected_set,
      kwiver::vital::image_container_sptr image )
{
  auto result = d->draw_detections( image, detected_set );
  return result;
}

} } } // end namespace
