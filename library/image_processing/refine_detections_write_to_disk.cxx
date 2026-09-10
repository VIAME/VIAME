// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of OCV refine detections draw debugging algorithm

#include "refine_detections_write_to_disk.h"

#include <algorithm>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <viame/algorithm_framework/exceptions/io.h>
#include <viame/algorithm_framework/util/string.h>
#include <viame/algorithm_framework/vital_config.h>

#include <kwiversys/SystemTools.hxx>

#include <viame/opencv_bridge/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace ocv {

using ST = kwiversys::SystemTools;

// Destructor
refine_detections_write_to_disk
::~refine_detections_write_to_disk()
{}

// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
refine_detections_write_to_disk
::check_configuration( [[maybe_unused]] vital::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
// Output images with tracked features drawn on them
vital::detected_object_set_sptr
refine_detections_write_to_disk
::refine(
  vital::image_container_sptr image_data,
  vital::detected_object_set_sptr detections ) const
{
  // Input validation and formatting
  this->frame_counter++;

  if( !detections )
  {
    return detections;
  }

  cv::Mat img = ocv::image_container::vital_to_ocv(
    image_data->get_image(),
    ocv::image_container::BGR_COLOR );

  // Get input filename if it's in the vital_metadata
  std::string filename;
  auto md = image_data->get_metadata();
  if( md )
  {
    if( auto& mdi = md->find( VITAL_META_IMAGE_URI ) )
    {
      // Get the full path, and then extract just the filename proper
      filename = ST::GetFilenameName( mdi.as_string() );
    }
  }

  for( auto det : *detections )
  {
    vital::bounding_box_d bbox = det->bounding_box();

    cv::Size s = img.size();
    vital::bounding_box_d bounds( vital::bounding_box_d::vector_type( 0, 0 ),
      vital::bounding_box_d::vector_type( s.width, s.height ) );

    // Clip detection box to image bounds.
    bbox = intersection( bounds, bbox );

    // Generate output filename. The pattern is a printf format string, so the
    // argument types have to line up with it exactly: two strings followed by
    // four ints. Passing anything else here is undefined behaviour.
    std::string category_str;
    std::string frame_str;

    if( !filename.empty() )
    {
      frame_str = filename;
    }
    else
    {
      // No source filename in the metadata, so fall back to a zero-padded
      // frame number.
      std::size_t const max_zeros = 6;
      frame_str = std::to_string( this->frame_counter );
      frame_str = std::string(
        max_zeros - std::min( max_zeros, frame_str.length() ), '0' ) +
        frame_str;
    }

    if( det->type() )
    {
      det->type()->get_most_likely( category_str );
    }
    if( !det->type() || category_str.empty() )
    {
      category_str = this->get_unknown_label();
    }

    std::string ofn = kwiver::vital::string_format(
      this->get_pattern(),
      category_str.c_str(),
      frame_str.c_str(),
      static_cast< int >( bbox.upper_left()[ 0 ] ),
      static_cast< int >( bbox.upper_left()[ 1 ] ),
      static_cast< int >( bbox.width() ),
      static_cast< int >( bbox.height() ) );

    this->detection_counter++;
    if( ofn.empty() )
    {
      LOG_ERROR(
        logger(),
        "Could not format output file name: \"" << this->get_pattern() <<
          "\"" );
      return detections;
    }

    // Output image to file
    // Make CV rect for out bbox coordinates
    cv::Rect r( bbox.upper_left()[ 0 ], bbox.upper_left()[ 1 ],
      bbox.width(), bbox.height() );

    cv::Mat crop = img( r );
    cv::imwrite( ofn, crop );
  } // end for

  return detections;
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver
