// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_FORMULATE_QUERY_H_
#define VITAL_ALGO_FORMULATE_QUERY_H_

#include <viame/algorithm_framework/vital_config.h>

#include <memory>
#include <string>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/descriptor_request.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/track_descriptor_set.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for formulating descriptors for queries
class VITAL_ALGO_EXPORT handle_descriptor_request
  : public kwiver::vital::algorithm
{
public:
  handle_descriptor_request();
  PLUGGABLE_INTERFACE( handle_descriptor_request );
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(
    kwiver::vital::config_block_sptr config ) const;

  /// Formulate query
  virtual bool handle(
    kwiver::vital::descriptor_request_sptr request,
    kwiver::vital::track_descriptor_set_sptr& desc,
    std::vector< kwiver::vital::image_container_sptr >& imgs ) = 0;
};

typedef std::shared_ptr< handle_descriptor_request >
  handle_descriptor_request_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_CONVERT_IMAGE_H_
