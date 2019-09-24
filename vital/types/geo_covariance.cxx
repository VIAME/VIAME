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
