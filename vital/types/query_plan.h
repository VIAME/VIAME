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
  ~query_plan() = default;

  /// Accessor for query plan unique identifier. \see set_id
  uid id() const;
  /// Accessor for query plan type. \see set_type
  query_type type() const;

  /// Accessor for temporal filter. \see set_temporal_filter
  filter temporal_filter() const;
  /// Accessor for temporal lower bound. \see set_temporal_bounds
  timestamp temporal_lower_bound() const;
  /// Accessor for temporal upper bound. \see set_temporal_bounds
  timestamp temporal_upper_bound() const;

  /// Accessor for spatial filter. \see set_spatial_filter
  filter spatial_filter() const;
  /// Accessor for spatial region. \see set_spatial_region
  geo_polygon spatial_region() const;

  /// Accessor for stream filter. \see set_stream_filter
  std::string stream_filter() const;

  /// Accessor for query descriptors. \see set_descriptors
  track_descriptor_set_sptr descriptors() const;
  /// Accessor for relevancy threshold. \see set_threshold
  double threshold() const;

  /**
   * \brief Set the query plan unique identifier.
   *
   * This sets the query plan unique identifier. Users should ensure that this
   * identifier uniquely identifies the query plan within the system. This
   * permits the system to recognize if the same query plan is reused. Once the
   * query plan has been seen by any component other than the original creator,
   * the identifier should be changed if the query plan is modified in any way.
   */
  void set_id( uid const& );
  void set_type( query_type );

  /**
   * \brief Set the temporal filter.
   *
   * This sets the temporal filter, which is used to decide how the temporal
   * bounds are applied to decide if a potential result is applicable to the
   * query.
   */
  void set_temporal_filter( filter );

  /**
   * \brief Set the temporal bounds.
   *
   * This sets the temporal bounds which are used, in conjunction with the
   * temporal filter, to limit the query results based on their temporal
   * locality.
   *
   * \throws std::logic_error Thrown if \p upper is less than \p lower.
   */
  void set_temporal_bounds( timestamp const& lower, timestamp const& upper );

  /**
   * \brief Set the spatial filter.
   *
   * This sets the spatial filter, which is used to decide how the spatial
   * region is applied to decide if a potential result is applicable to the
   * query.
   */
  void set_spatial_filter( filter );

  /**
   * \brief Set the spatial region.
   *
   * This sets the spatial region which is used, in conjunction with the
   * spatial filter, to limit the query results based on their spatial
   * locality.
   */
  void set_spatial_region( geo_polygon const& );

  /**
   * \brief Set the stream filter.
   *
   * This sets the stream filter, which is a string that the system uses to
   * decide if a potential result is applicable to the query based on the
   * source of the result's data.
   */
  void set_stream_filter( std::string const& );

  /**
   * \brief Set the query plan descriptors.
   *
   * This sets the descriptors that are used to select an initial set of
   * results. This applies only to similarity queries.
   */
  void set_descriptors( track_descriptor_set_sptr );

  /**
   * \brief Set the relevancy threshold.
   *
   * This sets the threshold that will be used to filter results based on their
   * relevancy score. This applies only to similarity queries.
   */
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
