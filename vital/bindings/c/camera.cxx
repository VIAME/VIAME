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
 * \brief vital::camera C interface implementation
 */

#include "camera.h"

#include <vital/types/camera.h>
#include <vital/io/camera_io.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/camera.h>

namespace kwiver {
namespace vital_c {

  SharedPointerCache< kwiver::vital::camera,
                    vital_camera_t > CAMERA_SPTR_CACHE( "camera" );

} }


/// Destroy a vital_camera_t instance
void vital_camera_destroy( vital_camera_t *cam,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera::destroy", eh,
    kwiver::vital_c::CAMERA_SPTR_CACHE.erase( cam );
  );
}


/// Read in a KRTD file, producing a new vital::camera object
vital_camera_t* vital_camera_read_krtd_file( char const *filepath,
                                             vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera::read_krtd_file", eh,
    kwiver::vital::camera_sptr c( kwiver::vital::read_krtd_file(filepath) );
    kwiver::vital_c::CAMERA_SPTR_CACHE.store( c );
    return reinterpret_cast<vital_camera_t*>( c.get() );
  );
  return 0;
}


/// Output the given vital_camera_t object to the specified file path
void vital_camera_write_krtd_file( vital_camera_t *cam,
                                   char const *filepath,
                                   vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera::write_krtd_file", eh,
    kwiver::vital::camera *m_cam = kwiver::vital_c::CAMERA_SPTR_CACHE.get( cam ).get();
    kwiver::vital::write_krtd_file( *m_cam,
                            filepath );
  );
}
