// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
