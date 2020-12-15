// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief This file contains the interface for a descriptor request.
 */

#ifndef VITAL_DESCRIPTOR_REQUEST_H_
#define VITAL_DESCRIPTOR_REQUEST_H_

#include "image_container.h"
#include "bounding_box.h"
#include "timestamp.h"
#include "track_descriptor.h"
#include "uid.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <memory>
#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/// A representation of a descriptor request.
///
/// This is used by some arbitrary GUI or other input to request and return
/// computed descriptors on some region of arbitrary input imagery.
class VITAL_EXPORT descriptor_request
{
public:

  descriptor_request();
  ~descriptor_request() = default;

  uid id() const;

  timestamp temporal_lower_bound() const;
  timestamp temporal_upper_bound() const;

  std::vector< bounding_box_i > spatial_regions() const;

  std::string data_location() const;
  std::vector< image_container_sptr> image_data() const;

  void set_id( uid const& );
  void set_temporal_bounds( timestamp const& lower, timestamp const& upper );
  void set_spatial_regions( std::vector< bounding_box_i > const& );

  void set_data_location( std::string const& );
  void set_image_data( std::vector< image_container_sptr > const& );

protected:

  vital::uid m_id;
  vital::timestamp m_temporal_lower;
  vital::timestamp m_temporal_upper;
  std::vector< bounding_box_i > m_spatial_regions;
  std::vector< image_container_sptr > m_image_data;
  std::string m_data_location;
};

/// Shared pointer for query plan
typedef std::shared_ptr< descriptor_request > descriptor_request_sptr;

} } // end namespace vital

#endif // VITAL_DESCRIPTOR_REQUEST_H_
