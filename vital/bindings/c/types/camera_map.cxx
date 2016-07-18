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
 * \brief C interface implementation for vital::camera_map
 */

#include "camera_map.h"

#include <vital/types/camera_map.h>
#include <vital/vital_foreach.h>

#include <vital/bindings/c/types/camera.h>
#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/camera.h>
#include <vital/bindings/c/helpers/camera_map.h>

namespace kwiver {
namespace vital_c {

SharedPointerCache<vital::camera_map,
                   vital_camera_map_t> CAM_MAP_SPTR_CACHE( "camera_map" );

}
}


using namespace kwiver;


/// New, simple camera map
vital_camera_map_t* vital_camera_map_new( size_t length,
                                          int64_t *frame_numbers,
                                          vital_camera_t **cameras,
                                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera_map::new", eh,
    if( frame_numbers == 0 || cameras == 0 )
    {
      length = 0;
    }

    // Create std::map of the paired items given
    // This assumes that the sptr type defined in the cache type is the same as
    // the type defined in C++-land (which is should be?)
    vital::camera_map::map_camera_t cmap;
    for( size_t i=0; i < length; i++ )
    {
      cmap[frame_numbers[i]] = vital_c::CAMERA_SPTR_CACHE.get( cameras[i] );
    }

    auto cm_sptr = std::make_shared< vital::simple_camera_map >( cmap );
    vital_c::CAM_MAP_SPTR_CACHE.store( cm_sptr );
    return reinterpret_cast<vital_camera_map_t*>( cm_sptr.get() );
  );
  return 0;
}


/// Destroy the given camera_map
void vital_camera_map_destroy( vital_camera_map_t *cam_map,
                               vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera_map::destroy", eh,
    vital_c::CAM_MAP_SPTR_CACHE.erase( cam_map );
  );
}


/// Return the number of cameras in the map
size_t vital_camera_map_size( vital_camera_map_t *cam_map,
                              vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera_map::size", eh,
    return vital_c::CAM_MAP_SPTR_CACHE.get( cam_map )->size();
  );
  return 0;
}


/// Set pointers to parallel arrays of frame numers and camera instances
void vital_camera_map_get_map( vital_camera_map_t *cam_map,
                               size_t *length,
                               int64_t **frame_numbers,
                               vital_camera_t ***cameras,
                               vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera_map::get_map", eh,
    vital::camera_map::map_camera_t map_cams =
      vital_c::CAM_MAP_SPTR_CACHE.get( cam_map )->cameras();

    *length = map_cams.size();
    *frame_numbers = (int64_t*)malloc(sizeof(int64_t) * *length);
    *cameras = (vital_camera_t**)malloc(sizeof(vital_camera_t*) * *length);
    size_t i=0;
    VITAL_FOREACH( vital::camera_map::map_camera_t::value_type const& p, map_cams )
    {
      (*frame_numbers)[i] = p.first;
      vital_c::CAMERA_SPTR_CACHE.store( p.second );
      (*cameras)[i] = reinterpret_cast< vital_camera_t* >( p.second.get() );
      ++i;
    }
  );
}
