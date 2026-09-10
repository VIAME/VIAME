// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining the handle_descriptor_request_core algorithm

#ifndef ARROWS_PLUGINS_CORE_FORMULATE_QUERY_CORE_H_
#define ARROWS_PLUGINS_CORE_FORMULATE_QUERY_CORE_H_

#include "viame_descriptors_export.h"

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>
#include <viame/algorithm_framework/algo/handle_descriptor_request.h>

#include <viame/algorithm_framework/algo/compute_track_descriptors.h>
#include <viame/algorithm_framework/algo/image_io.h>

namespace kwiver {

namespace arrows {

namespace core {

/// A basic query formulator
class VIAME_DESCRIPTORS_EXPORT handle_descriptor_request_core
  : public vital::algo::handle_descriptor_request
{
public:
  PLUGGABLE_IMPL(
    handle_descriptor_request_core,
    "Formulate descriptors for later queries.",
    PARAM(
      image_reader, vital::algo::image_io_sptr,
      "image_reader" ),
    PARAM(
      descriptor_extractor, vital::algo::compute_track_descriptors_sptr,
      "descriptor_extractor" )
  )

  /// Destructor
  virtual ~handle_descriptor_request_core() = default;

  /// Check that the algorithm's currently configuration is valid
  ///
  /// This checks solely within the provided \c config_block and not against
  /// the current state of the instance. This isn't static for inheritence
  /// reasons.
  ///
  /// \param config  The config block to check configuration of.
  ///
  /// \returns true if the configuration check passed and false if it didn't.
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Formulate query
  virtual bool handle(
    kwiver::vital::descriptor_request_sptr request,
    kwiver::vital::track_descriptor_set_sptr& desc,
    std::vector< kwiver::vital::image_container_sptr >& imgs );

private:
  void initialize() override;
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif
