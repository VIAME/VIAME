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

using namespace viame;

namespace viame {

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
::check_configuration( viame::config_block_sptr config ) const
{
  return (
    viame::check_nested_algo_configuration< algo::image_io >(
      "image_reader", config )
    &&
    viame::check_nested_algo_configuration< algo::compute_track_descriptors >(
      "descriptor_extractor", config )
  );
}

/// Extend a previous set of tracks using the current frame
bool
handle_descriptor_request_core
::handle(
  viame::descriptor_request_sptr request,
  viame::track_descriptor_set_sptr& descs,
  std::vector< viame::image_container_sptr >& imgs )
{
  // Verify that all dependent algorithms have been initialized
  if( !c_image_reader || !c_descriptor_extractor )
  {
    // Something did not initialize
    VITAL_THROW(
      viame::algorithm_configuration_exception, this->interface_name(),
      this->plugin_name(),
      "not all sub-algorithms have been initialized" );
  }

  // load images or video if required by query plan
  std::string data_path = request->data_location();
  viame::image_container_sptr image = c_image_reader->load( data_path );

  if( !image )
  {
    throw std::runtime_error( "Handler unable to load image" );
  }

  // extract descriptors on the current frame
  viame::timestamp fake_ts( 0, 0 );
  viame::track_sptr ff_track = viame::track::create();
  ff_track->set_id( 0 );

  viame::bounding_box_d dims( 0, 0, image->width(), image->height() );

  viame::detected_object_sptr det(
    new viame::detected_object( dims ) );
  viame::track_state_sptr state1(
    new viame::object_track_state( fake_ts, det ) );

  ff_track->append( state1 );

  std::vector< viame::track_sptr > trk_vec;
  trk_vec.push_back( ff_track );

  viame::object_track_set_sptr tracks(
    new viame::object_track_set( trk_vec ) );

  descs = c_descriptor_extractor->compute( fake_ts, image, tracks );

  imgs.clear();
  imgs.push_back( image );
  return true;
}

} // namespace core

} // namespace viame
