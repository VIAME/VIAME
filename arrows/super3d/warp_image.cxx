// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

 /**
 * \file
 * \brief Source file for warp_image
 */

#include <vgl/algo/vgl_h_matrix_2d.h>

namespace kwiver {
namespace arrows {
namespace super3d {

bool
warp_image_is_identity( vgl_h_matrix_2d<double> const& H )
{
  vnl_matrix_fixed<double,3,3> const& M = H.get_matrix();
  return ( M(0,1) == 0.0 && M(0,2) == 0.0 &&
           M(1,0) == 0.0 && M(1,2) == 0.0 &&
           M(2,0) == 0.0 && M(2,1) == 0.0 &&
           M(0,0) == M(1,1) && M(1,1) == M(2,2) );
}

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
