// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "warp_image.hxx"
#include <vil/vil_bicub_interp.hxx>

#define INSTANTIATE( PixType )                                              \
template bool kwiver::arrows::super3d::warp_image( vil_image_view<PixType> const& src,        \
                                 vil_image_view<PixType>& dest,             \
                                 vgl_h_matrix_2d<double> const& dest_to_src_homography, \
                                 vil_image_view< bool > * const unmapped_mask );  \
                                                                            \
template bool kwiver::arrows::super3d::warp_image( vil_image_view<PixType> const& src,        \
                                 vil_image_view<PixType>& dest,             \
                                 vgl_h_matrix_2d<double> const& dest_to_src_homography, \
                                 int, int,                                  \
                                 vil_image_view< bool > * const unmapped_mask );  \
                                                                            \
template bool kwiver::arrows::super3d::warp_image( vil_image_view<PixType> const& src,        \
                                 vil_image_view<PixType>& dest,             \
                                 vgl_h_matrix_2d<double> const& dest_to_src_homography, \
                                 warp_image_parameters const& param,        \
                                 vil_image_view< bool > * const unmapped_mask );  \

INSTANTIATE( bool );
INSTANTIATE( vxl_byte );
INSTANTIATE( vxl_uint_16 );
INSTANTIATE( double );
INSTANTIATE( float )

VIL_BICUB_INTERP_INSTANTIATE( bool );
