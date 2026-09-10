// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of handle_descriptor_request_core

#include "handle_descriptor_request_core.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/exceptions/algorithm.h>
#include <viame/algorithm_framework/exceptions/image.h>
#include <viame/core_types/descriptor_request.h>

using namespace kwiver::vital;

namespace kwiver {

namespace arrows {

namespace core {

// ----------------------------------------------------------------------------
void
handle_descriptor_request_core
::initialize()
{
  attach_logger( "arrows.core.handle_descriptor_request_core" );
}

bool
handle_descriptor_request_core
::check_configuration( vital::config_block_sptr config ) const
{
  return (
    kwiver::vital::check_nested_algo_configuration< algo::image_io >(
      "image_reader", config )
    &&
    kwiver::vital::check_nested_algo_configuration< algo::compute_track_descriptors >(
      "descriptor_extractor", config )
  );
}

/// Extend a previous set of tracks using the current frame
bool
handle_descriptor_request_core
::handle(
  kwiver::vital::descriptor_request_sptr request,
  kwiver::vital::track_descriptor_set_sptr& descs,
  std::vector< kwiver::vital::image_container_sptr >& imgs )
{
  // Verify that all dependent algorithms have been initialized
  if( !c_image_reader || !c_descriptor_extractor )
  {
    // Something did not initialize
    VITAL_THROW(
      vital::algorithm_configuration_exception, this->interface_name(),
      this->plugin_name(),
      "not all sub-algorithms have been initialized" );
  }

  // load images or video if required by query plan
  std::string data_path = request->data_location();
  kwiver::vital::image_container_sptr image = c_image_reader->load( data_path );

  if( !image )
  {
    throw std::runtime_error( "Handler unable to load image" );
  }

  // extract descriptors on the current frame
  vital::timestamp fake_ts( 0, 0 );
  vital::track_sptr ff_track = vital::track::create();
  ff_track->set_id( 0 );

  vital::bounding_box_d dims( 0, 0, image->width(), image->height() );

  vital::detected_object_sptr det(
    new vital::detected_object( dims ) );
  vital::track_state_sptr state1(
    new vital::object_track_state( fake_ts, det ) );

  ff_track->append( state1 );

  std::vector< vital::track_sptr > trk_vec;
  trk_vec.push_back( ff_track );

  vital::object_track_set_sptr tracks(
    new vital::object_track_set( trk_vec ) );

  descs = c_descriptor_extractor->compute( fake_ts, image, tracks );

  imgs.clear();
  imgs.push_back( image );
  return true;
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
