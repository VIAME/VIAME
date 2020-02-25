/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
  * \brief This file contains the interface to a geo point.
  */

#ifndef KWIVER_VITAL_GEO_COVARIANCE_H_
#define KWIVER_VITAL_GEO_COVARIANCE_H_

#include <vital/types/geo_point.h>
#include <vital/types/covariance.h>
#include <memory>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/** A geo_point with covariance
 */
class VITAL_EXPORT geo_covariance : public geo_point
{
public:
  using covariance_type = covariance_<3, float>;

  geo_covariance();
  geo_covariance( geo_2d_point_t const&, int crs );
  geo_covariance( geo_3d_point_t const& pt, int crs );

  virtual ~geo_covariance() = default;

  covariance_type covariance() const { return m_covariance; }
  void set_covariance( covariance_3f c )
  {
    m_covariance = c;
  }

protected:

  covariance_type m_covariance;
};

VITAL_EXPORT::std::ostream& operator<< ( ::std::ostream& str, geo_covariance const& obj );

}
} // end namespace

#endif
