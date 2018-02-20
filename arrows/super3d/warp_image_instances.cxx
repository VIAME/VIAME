/*ckwg +29
 * Copyright 2010-2016 by Kitware, Inc.
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
