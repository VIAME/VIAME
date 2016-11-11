/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief core polygon interface
 */

#ifndef KWIVER_ALGORITHM_CORE_POLYGON_H
#define KWIVER_ALGORITHM_CORE_POLYGON_H

#include <vital/vital_config.h>
#include <arrows/core/kwiver_algo_export.h>

#include <vital/types/polygon.h>

#include <vector>
#include <memory>

namespace kwiver {
namespace arrows {
namespace core {

class polygon;
typedef std::shared_ptr< polygon > polygon_sptr;

// ----------------------------------------------------------------
/**
 * @brief Basic polygon implementation.
 *
 * This class represents a very basic polygon implementation that
 * implements the base class interface.
 *
 * This implementation of a polygon is considered the basic polygon
 * that can be converted from/to any of the other concrete
 * implementations.
 */
class KWIVER_ALGO_EXPORT polygon
  : public kwiver::vital::polygon
{
public:
  polygon();
  polygon( const std::vector< kwiver::vital::polygon::point_t >& dat);
  virtual ~polygon();

  virtual void push_back( double x, double y );
  virtual void push_back( const kwiver::vital::polygon::point_t& pt );
  virtual size_t num_vertices() const;
  virtual bool contains( double x, double y );
  virtual bool contains( const kwiver::vital::polygon::point_t& pt );
  virtual kwiver::vital::polygon::point_t at( size_t idx ) const;

  /**
   * @brief Get list of vertices.
   *
   * This method returns the list of points that make up the polygon.
   *
   * @return List of vertices.
   */
  const std::vector< kwiver::vital::polygon::point_t >& get_polygon() const;

private:
  std::vector< kwiver::vital::polygon::point_t > m_polygon;

}; // end class polygon

} } } // end namespace

#endif // KWIVER_ALGORITHM_CORE_POLYGON_H
