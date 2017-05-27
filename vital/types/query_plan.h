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
 * \brief This file contains the interface to a query plan.
 */

#ifndef VITAL_QUERY_PLAN_H_
#define VITAL_QUERY_PLAN_H_

#include "geo_polygon.h"
#include "timestamp.h"
#include "track_descriptor_set.h"
#include "uid.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <memory>

namespace kwiver {
namespace vital {

enum class filter
{
  IGNORE = 0,
  CONTAINS_WHOLLY,
  CONTAINS_PARTLY,
  INTERSECTS, // partly but not wholly contained
  INTERSECTS_INBOUND, // first does not contain, then contains
  INTERSECTS_OUTBOUND, // first contains, then does not contain
  DOES_NOT_CONTAIN,
};

// ----------------------------------------------------------------------------
/// A representation of a query plan.
class query_plan
{
public:
  enum query_type
  {
    SIMILARITY = 0,
    // TODO add other types
  };

  query_plan();
  ~query_plan() VITAL_DEFAULT_DTOR

  uid id() const;
  query_type type() const;

  filter temporal_filter() const;
  timestamp temporal_lower_bound() const;
  timestamp temporal_upper_bound() const;

  filter spatial_filter() const;
  geo_polygon spatial_region() const;

  std::string stream_filter() const;

  track_descriptor_set_sptr descriptors() const;
  double threshold() const;

  void set_id( uid const& );
  void set_type( query_type );

  void set_temporal_filter( filter );
  void set_temporal_bounds( timestamp const& lower, timestamp const& upper );

  void set_spatial_filter( filter );
  void set_spatial_region( geo_polygon const& );

  void set_stream_filter( std::string const& );

  void set_descriptors( track_descriptor_set_sptr );
  void set_threshold( double );

protected:

  uid m_id;
  query_type m_type;
  filter m_temporal_filter;
  timestamp m_temporal_lower;
  timestamp m_temporal_upper;
  filter m_spatial_filter;
  geo_polygon m_spatial_region;
  std::string m_stream_filter;
  track_descriptor_set_sptr m_descriptors;
  double m_threshold;


};

/// Shared pointer for query plan
typedef std::shared_ptr< query_plan > query_plan_sptr;

} } // end namespace vital

#endif // VITAL_QUERY_PLAN_H_
