// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for draw_detected_object_set

#include "draw_detected_object_set.h"

#include <viame/algorithm_framework/config/config_difference.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/algorithm_framework/util/tokenize.h>
#include <viame/algorithm_framework/vital_config.h>
#include <viame/core_types/vector.h>
#include <viame/core_types/vital_types.h>

#include <image_ops/channels.h>
#include <image_ops/draw.h>
#include <image_ops/filter.h>

#include <viame/core_types/image_container.h>
#include <kwiversys/RegularExpression.hxx>


#include <sstream>

namespace kwiver {

namespace arrows {

namespace ocv {

// Constant for offsetting drawn labels
static const int MULTI_LABEL_OFFSET( 15 );

typedef  kwiver::vital::vector_< 3, unsigned int > ColorVector;

namespace io = viame::image_ops;
namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
/// @brief
///
class draw_detected_object_set::priv
{
public:
  // -- CONSTRUCTORS --
  priv( draw_detected_object_set& parent )
    : m_parent( parent ),
      m_config_error( false )
  {
    m_default_params.thickness = 1.0;
    m_default_params.color[ 0 ] = 255;
    m_default_params.color[ 1 ] = 0;
    m_default_params.color[ 2 ] = 0;
  }

  ~priv()
  {}

  draw_detected_object_set& m_parent;

  // internal state
  bool m_config_error;

  // Configuration values
  float
  m_threshold() const { return m_parent.get_threshold(); }

  std::vector< std::string > m_select_classes;
  bool
  m_do_alpha() const { return m_parent.get_alpha_blend_prob(); }

  struct Bound_Box_Params
  {
    float thickness;
    ColorVector color;
  } m_default_params;

  // box attributes per object type
  std::map< std::string, Bound_Box_Params > m_custum_colors;

  float
  m_text_scale() const { return m_parent.get_text_scale(); }
  float
  m_text_thickness() const { return m_parent.get_text_thickness(); }
  bool
  m_clip_box_to_image() const { return m_parent.get_clip_box_to_image(); }
  bool
  m_draw_text() const { return m_parent.get_draw_text(); }

  // --------------------------------------------------------------------------
  /// @brief Draw a box on an image.
  ///
  /// This method draws a box on an image for the bounding box from a
  /// detected object.
  ///
  /// When drawing a box with multiple class names, draw the first
  /// class_name with the \c just_text parameter \b false and all
  /// subsequent calls with it set to \b true. Also the \c offset
  /// parameter must be incremented so the labels do not overwrite.
  ///
  /// @param[in,out] image Input image updated with drawn box
  /// @param[in] dos detected object with bounding box
  /// @param[in] label Text label to use for box
  /// @param[in] prob Probability value to add to label text
  /// @param[in] just_text Set to true if only draw text, not the
  ///            bounding box. This is used when there are multiple
  ///            labels for the same detection.
  /// @param[in] offset How much to offset text fill box from text
  ///            baseline. This is used to offset labels when there are
  ///            more than one label for a detection.
  void
  draw_box(
    kv::image_of< uint8_t >&     image,
    const vital::detected_object_sptr dos,
    std::string label,
    double prob,
    bool just_text = false,
    int offset_index = 0 ) const
  {
    // A copy to draw on, blended back at the end so that the alpha shading
    // is over the frame rather than over the previous box. `vital::image`
    // copies share their memory, so this is a deep one.
    kv::image_of< uint8_t > overlay( image.width(), image.height(),
                                     image.depth() );

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        for( size_t p = 0; p < image.depth(); ++p )
        {
          overlay( i, j, p ) = image( i, j, p );
        }
      }
    }

    vital::bounding_box_d bbox = dos->bounding_box();
    if( m_clip_box_to_image() )
    {
      vital::bounding_box_d img(
        vital::bounding_box_d::vector_type( 0, 0 ),
        vital::bounding_box_d::vector_type(
          static_cast< double >( image.width() ),
          static_cast< double >( image.height() ) ) );
      bbox = intersection( img, bbox );
    }

    auto const left = static_cast< long >( bbox.upper_left()[ 0 ] );
    auto const top = static_cast< long >( bbox.upper_left()[ 1 ] );

    io::rect const r{ left, top,
                      left + static_cast< long >( bbox.width() ),
                      top + static_cast< long >( bbox.height() ) };
    std::string p = std::to_string( static_cast< long double >( prob ) ); // convert
                                                                          // value
                                                                          // to
                                                                          // string
    std::string txt = label + " " + p;

    // Clip threshold to limit value. If less than 0.05, leave threshold as it
    // is.
    // Else lower by 5%. This is a heuristic for making the alpha shading look
    // good.
    double tmp_thresh = ( this->m_threshold() - ( ( this->m_threshold() >=
                                                    0.05 ) ? 0.05 : 0 ) );

    double alpha_wight = ( m_do_alpha() ) ? ( ( prob - tmp_thresh ) /
                                              ( 1 - tmp_thresh ) ) : 1.0;

    Bound_Box_Params const* bbp = &m_default_params;
    auto iter = m_custum_colors.find( label );

    // look for custom color for this class_name
    if( iter != m_custum_colors.end() )
    {
      bbp = &( iter->second );
    }

    // Add text to an existing box
    if( !just_text )
    {
      // Reversed on purpose. The config calls this triple RGB and its
      // default is "0 0 255", but it was handed to `cv::Scalar` over a BGR
      // image, so what was actually drawn was (B, G, R) -- the default drew
      // a *red* box while the documentation said blue, and the example in
      // `custom_class_color` said the opposite of what it did.
      //
      // Reproducing that here keeps every existing pipeline and every
      // screenshot the same. Correcting the documentation instead of the
      // code is a decision for someone who knows which users are relying on
      // which; see lite-findings.md 1.10.
      io::colour const colour{ static_cast< double >( bbp->color[ 2 ] ),
                               static_cast< double >( bbp->color[ 1 ] ),
                               static_cast< double >( bbp->color[ 0 ] ) };

      io::draw_rect( overlay, r, colour,
                     static_cast< long >( bbp->thickness ) );
    }

    if( m_draw_text() )
    {
      // The text scale was a fraction for Hershey's vector font and is a
      // whole multiple for the bitmap one, so it is rounded up rather than
      // truncated: a configured 0.5 means "small", and the smallest this
      // font goes is one.
      auto const scale = std::max(
        1L, static_cast< long >( std::lround( m_text_scale() ) ) );

      auto const pen_i = r.left;
      auto const pen_j = r.top + MULTI_LABEL_OFFSET * offset_index;

      auto const size = io::text_size( txt, scale );

      // A filled black plate behind the label, as before. Placed from the
      // pen downwards rather than up from a baseline: `draw_text` takes the
      // top left, since a bitmap font has no baseline to hang from.
      io::draw_rect( overlay,
                     io::rect{ pen_i, pen_j, pen_i + size.right,
                               pen_j + size.bottom },
                     io::colour{ 0.0, 0.0, 0.0 }, -1 );

      io::draw_text( overlay, txt, pen_i, pen_j,
                     io::colour{ 255.0, 255.0, 255.0 }, scale );
    }

    image = io::add_weighted( overlay, alpha_wight, image,
                              1.0 - alpha_wight );
  } // draw_box

  // --------------------------------------------------------------------------
  /// @brief Draw detected object on image.
  ///
  /// This method draws the detections on a copy of the supplied
  /// image. The detections are drawn in confidence order up to the
  /// threshold. For each detection, the most likely class_name is
  /// optionally displayed below the box.
  ///
  /// @param image_data The image to draw on.
  /// @param input_set List of detections to draw.
  ///
  /// @return New image with boxes drawn.
  vital::image_container_sptr
  draw_detections(
    vital::image_container_sptr image_data,
    vital::detected_object_set_sptr in_set ) const
  {
    // Three planes and eight bit, whatever came in: the overlay is drawn in
    // colour and a grey frame would have nowhere to put it.
    kv::image_of< uint8_t > image =
      io::force_three_channels( kv::image_of< uint8_t >(
        image_data->get_image() ) );

    // process the detection set
    auto ie = in_set->cend();
    for( auto det = in_set->cbegin(); det != ie; ++det )
    {
      auto det_type = ( *det )->type();
      if( !det_type || det_type->size() == 0 )
      {
        // No type has been assigned. Just filter on threshold
        if( ( *det )->confidence() < m_threshold() )
        {
          continue;
        }

        draw_box( image, *det, "", ( *det )->confidence() );
        continue;
      }

      // ----------------------------------------------------------------------
      // Since there is a type assigned, select on specified class_names
      auto names = det_type->class_names(); // get all class_names

      bool text_only( false );
      int count( 0 );

      // Draw once for each selected class_name
      for( auto n : names )
      {
        double score = det_type->score( n );
        if( score < m_threshold() || !name_selected( n ) )
        {
          continue;
        }

        LOG_TRACE(
          m_parent.logger(),
          "Drawing box for class: " << n << "   score: " << score );
        draw_box( image, *det, n, score, text_only, count );
        text_only = true; // skip box on all subsequent calls
      }
    } // end foreach

    return std::make_shared< vital::simple_image_container >(
      vital::image( image ) );
  } // end draw_detections

// ----------------------------------------------------------------------------
  /// @brief See if name has been selected for display.
  ///
  /// @param name Name to check.
  ///
  /// @return \b true if name should be rendered
  bool
  name_selected( std::string const& name ) const
  {
    if( m_select_classes[ 0 ] == "*ALL*" )
    {
      return true;
    }

    return ( std::find(
      m_select_classes.begin(), m_select_classes.end(),
      name ) != m_select_classes.end() );
  }

// ----------------------------------------------------------------------------
  void
  process_config()
  {
    // Parse custom class color specification
    // class/line-thickness/color-rgb;class/line-thickness/color-rgb
    // e.g. person/3.5/0 0 255;
    {
      std::vector< std::string > cspec;
      kwiver::vital::tokenize(
        m_parent.c_custom_class_color, cspec, ";",
        kwiver::vital::TokenizeTrimEmpty );

      for( auto cs : cspec )
      {
        kwiversys::RegularExpression exp(
          "\\$([^/]+)/([0-9.]+)/([0-9]+) ([0-9]+) ([0-9]+)" );

        if( !exp.find( cs ) )
        {
          // parse error - log something
          m_config_error = true;
          LOG_ERROR(
            m_parent.logger(),
            "Error parsing custom color specification \"" << cs << "\"" );

          return;
        }

        // exp.match(0) - whole match
        // exp.match(1) - class_name string
        // exp.match(2) - line thickness
        // exp.match(3) - color red
        // exp.match(4) - color green
        // exp.match(5) - color blue

        draw_detected_object_set::priv::Bound_Box_Params bp;

        bp.thickness = std::stof( exp.match( 2 ) );
        bp.color[ 0 ] = std::stoi( exp.match( 5 ) );
        bp.color[ 1 ] = std::stoi( exp.match( 4 ) );
        bp.color[ 2 ] = std::stoi( exp.match( 3 ) );

        m_custum_colors[ exp.match( 1 ) ] = bp; // add to map
      } // end foreach
    } // end local scope

    {
      // parse defaults default color
      kwiversys::RegularExpression exp( "([0-9]+) ([0-9]+) ([0-9]+)" );

      if( !exp.find( m_parent.c_default_color ) )
      {
        // parse error - log something
        m_config_error = true;
        LOG_ERROR(
          m_parent.logger(), "Error parsing custom color specification \""
            << m_parent.c_default_color << "\"" );
        return;
      }

      // exp.match(0) - whole match
      // exp.match(1) - color red
      // exp.match(2) - color green
      // exp.match(3) - color blue

      m_default_params.color[ 0 ] = std::stoi( exp.match( 3 ) );
      m_default_params.color[ 1 ] = std::stoi( exp.match( 2 ) );
      m_default_params.color[ 2 ] = std::stoi( exp.match( 1 ) );
    } // end local scope

    // Parse selected class_names
    kwiver::vital::tokenize(
      m_parent.c_select_classes, m_select_classes, ";",
      kwiver::vital::TokenizeTrimEmpty );
  }
}; // end priv class

// ----------------------------------------------------------------------------
void
draw_detected_object_set
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
}

draw_detected_object_set::
~draw_detected_object_set()
{}

void
draw_detected_object_set
::set_configuration_internal( vital::config_block_sptr in_config )
{
  vital::config_block_sptr config = this->get_configuration();
  kwiver::vital::config_difference cd( config, in_config );
  cd.warn_extra_keys( logger() );

  d->process_config();
}

// ----------------------------------------------------------------------------
bool
draw_detected_object_set
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  // This can be called before the config is "set". A more robust way
  // of determining validity should be used.
  return !d->m_config_error;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
draw_detected_object_set
::draw(
  kwiver::vital::detected_object_set_sptr detected_set,
  kwiver::vital::image_container_sptr image )
{
  //  Update config to get the latest values which could be set via setters
  d->process_config();

  auto result = d->draw_detections( image, detected_set );
  return result;
}

} // namespace ocv

} // namespace arrows

}     // end namespace
