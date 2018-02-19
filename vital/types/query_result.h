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
 * \brief This file contains the interface for a query result.
 */

#ifndef VITAL_QUERY_RESULT_H_
#define VITAL_QUERY_RESULT_H_

#include "image_container.h"
#include "timestamp.h"
#include "object_track_set.h"
#include "track_descriptor_set.h"
#include "track_descriptor.h"
#include "geo_point.h"
#include "uid.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <memory>
#include <string>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/// A representation of a database query result.
///
/// This structure is used as the response to a query, for communication with
/// either a GUI or other entity. It contains many optional fields which only
/// need be filled based on the application.
class VITAL_EXPORT query_result
{
public:

  query_result();
  ~query_result() VITAL_DEFAULT_DTOR

  uid query_id() const;
  std::string stream_id() const;

  unsigned instance_id() const;
  double relevancy_score() const;

  vital::timestamp start_time() const;
  vital::timestamp end_time() const;

  vital::geo_point location() const;

  object_track_set_sptr tracks() const;
  track_descriptor_set_sptr descriptors() const;

  std::vector< image_container_sptr > image_data() const;

  void set_query_id( uid const& );
  void set_stream_id( std::string const& );

  void set_instance_id( unsigned );
  void set_relevancy_score( double );

  void set_temporal_bounds( timestamp const&, timestamp const& );

  void set_location( vital::geo_point );

  void set_tracks( object_track_set_sptr );
  void set_descriptors( track_descriptor_set_sptr );

  void set_image_data( std::vector< image_container_sptr > const& );

protected:

  vital::uid m_query_id;
  std::string m_stream_id;

  unsigned m_instance_id;
  double m_relevancy_score;

  vital::timestamp m_start_time;
  vital::timestamp m_end_time;

  vital::geo_point m_location;

  object_track_set_sptr m_tracks;
  track_descriptor_set_sptr m_descriptors;

  std::vector< image_container_sptr > m_image_data;
};

/// Shared pointer for query result
typedef std::shared_ptr< query_result > query_result_sptr;

} } // end namespace vital

#endif // VITAL_QUERY_RESULT_H_
