/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
  ~descriptor_request() VITAL_DEFAULT_DTOR

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
