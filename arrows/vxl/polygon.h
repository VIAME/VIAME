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
 * \brief vxl polygon interface
 */

#ifndef KWIVER_ALGORITHM_VXL_POLYGON_H
#define KWIVER_ALGORITHM_VXL_POLYGON_H

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/types/polygon.h>

#include <vgl/vgl_polygon.h>

namespace kwiver {
namespace arrows {
namespace vxl {

class polygon;
typedef std::shared_ptr< polygon > polygon_sptr;

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class KWIVER_ALGO_VXL_EXPORT polygon
  : public kwiver::vital::polygon
{
public:
  // -- CONSTRUCTORS --
  polygon();
  explicit polygon( const vgl_polygon< double >& poly );
  virtual ~polygon();

  virtual void push_back( double x, double y );
  virtual void push_back( const point_t& pt );
  virtual size_t num_vertices() const;
  virtual std::vector< kwiver::vital::polygon::point_t > get_vertices() const;
  virtual bool contains( double x, double y );
  virtual bool contains( const point_t& pt );
  virtual kwiver::vital::polygon::point_t at( size_t idx ) const;
  virtual kwiver::vital::vital_polygon_sptr get_polygon();

  // Allows access and modification to the implementation polygon.
  vgl_polygon<double>& get_vgl_polygon() { return m_polygon; }

  /**
   * @brief Convert abstract polygon to vxl polygon.
   *
   * This static method down-casts/converts a generic polygon into a
   * vxl::polygon.  If the generic polygon is really a vxl::polygon
   * type, then a managed pointer of the correct type to the input
   * object is returned. If the generic polygon is not a vxl::polygon
   * then the data is converted to a vxl::polygon and a new object is
   * returned.
   *
   * @param poly Abstract polygon
   *
   * @return Polygon of this type.
   */
  static kwiver::arrows::vxl::polygon_sptr get_vxl_polygon( kwiver::vital::polygon_sptr poly );

private:
  vgl_polygon< double > m_polygon;

}; // end class vxl_polygon

} } } // end namespace

#endif /* KWIVER_ALGORITHM_VXL_POLYGON_H */
