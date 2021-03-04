// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

 /**
  * \file
  * \brief This file contains the implementation of a geo covariant point.
  */

#include "geo_covariance.h"

#include <iomanip>
#include <stdexcept>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
geo_covariance::
geo_covariance()
  : geo_point()
{ }

// ----------------------------------------------------------------------------
geo_covariance::
geo_covariance( geo_2d_point_t const& point, int crs )
  : geo_point( point, crs )
{

}

// ----------------------------------------------------------------------------
geo_covariance::
geo_covariance( geo_3d_point_t const& point, int crs )
  : geo_point(point, crs)
{

}

  // ----------------------------------------------------------------------------
std::ostream&
operator<<( std::ostream& str, vital::geo_covariance const& obj )
{
  str << "geo_covariance\n";

  if (obj.is_empty())
  {
    str << "[ empty ]";
  }
  else
  {
    auto const old_prec = str.precision();
    auto const loc = obj.location();

    str << " - value : ";
    str << std::setprecision(22)
      << "[ " << loc[0]
      << ", " << loc[1]
      << ", " << loc[2]
      << " ] @ " << obj.crs() << "\n";

    str << " - covariance  : ";
    auto const c = obj.covariance().matrix();
    str << std::setprecision(22)
      << "[ " << c(0, 0) << ", " << c(0, 1) << ", " << c(0, 2) << "\n                   "
      << c(1, 0) << ", " << c(1, 1) << ", " << c(1, 2) << "\n                   "
      << c(2, 0) << ", " << c(2, 1) << ", " << c(2, 2) << " ]\n";

    str.precision(old_prec);
  }

  return str;
}

}
} // end namespace
