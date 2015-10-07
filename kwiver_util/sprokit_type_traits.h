/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#ifndef _VITAL_TYPES_VITAL_H_
#define _VITAL_TYPES_VITAL_H_

#include <vital/vital_types.h>
#include <vital/types/geo_lat_lon.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/track_set.h>

#include <kwiver_util/corner_points.h>

#include <kwiver_util/trait_utils.h>

namespace kwiver {
namespace vital {

  class timestamp;
  class f2f_homography;

} }


// ================================================================
//
// Create type traits for common pipeline types.
// ( type-trait-name, "canonical_type_name", concrete-type )
//
create_type_trait( timestamp, "kwiver:timestamp", kwiver::vital::timestamp );
create_type_trait( gsd, "kwiver:gsd", kwiver::vital::gsd_t );
create_type_trait( corner_points, "corner_points", kwiver::vital::corner_points_t );
create_type_trait( image, "kwiver:image_container", kwiver::vital::image_container_sptr ); // polymorphic type must pass by reference
create_type_trait( feature_set, "kwiver:feature_set", kwiver::vital::feature_set_sptr );
create_type_trait( descriptor_set, "kwiver:descriptor_set", kwiver::vital::descriptor_set_sptr );
create_type_trait( track_set, "kwiver:track_set", kwiver::vital::track_set_sptr );

create_type_trait( homography_src_to_ref, "kwiver:s2r_homography", kwiver::vital::f2f_homography );
create_type_trait( homography_ref_to_src, "kwiver:r2s_homography", kwiver::vital::f2f_homography );
create_type_trait( image_file_name, "kwiver:image_file_name", kwiver::vital::path_t );
create_type_trait( video_file_name, "kwiver:video_file_name", kwiver::vital::path_t );


// ================================================================
//
// Create port traits for common port types.
// ( port-name, type-trait-name, "port-description" )
//
create_port_trait( timestamp, timestamp, "Timestamp for input image." );
create_port_trait( corner_points, corner_points, "Four corner points for image in lat/lon units, ordering ul ur lr ll." );
create_port_trait( gsd, gsd, "GSD for image in meters per pixel." );
create_port_trait( image, image, "Single frame image." );
create_port_trait( feature_set, feature_set, "Set of detected image features" );
create_port_trait( descriptor_set, descriptor_set, "Set of feature descriptors" );
create_port_trait( track_set, track_set, "Set of feature tracks for stabilization" );

create_port_trait( homography_src_to_ref, homography_src_to_ref, "Source image to ref image homography." );
create_port_trait( image_file_name, image_file_name, "Name of an image file. Usually a single frame of a video." );
create_port_trait( video_file_name, video_file_name, "Name of video file." );

#endif /* _KWIVER_TYPES_KWIVER_H_ */
