// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_ALGO_FORMULATE_QUERY_H_
#define VITAL_ALGO_FORMULATE_QUERY_H_

#include <vital/vital_config.h>

#include <string>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/track_descriptor_set.h>
#include <vital/types/descriptor_request.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for formulating descriptors for queries
class VITAL_ALGO_EXPORT handle_descriptor_request
  : public kwiver::vital::algorithm_def<handle_descriptor_request>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "handle_descriptor_request"; }

  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Formulate query
  virtual bool handle(
    kwiver::vital::descriptor_request_sptr request,
    kwiver::vital::track_descriptor_set_sptr& desc,
    std::vector< kwiver::vital::image_container_sptr >& imgs ) = 0;

protected:
  handle_descriptor_request();

};

typedef std::shared_ptr<handle_descriptor_request> handle_descriptor_request_sptr;

} } } // end namespace

#endif // VITAL_ALGO_CONVERT_IMAGE_H_
