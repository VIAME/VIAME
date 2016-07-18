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

#include <vital/io/camera_io.h>
#include <vital/types/camera.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/camera.h>
#include <vital/bindings/c/helpers/camera_intrinsics.h>


namespace kwiver {
namespace vital_c {

  SharedPointerCache< vital::camera,
                      vital_camera_t > CAMERA_SPTR_CACHE( "camera" );

}
}


using namespace kwiver;


/// Destroy a vital_camera_t instance
void vital_camera_destroy( vital_camera_t *cam,
                           vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera::destroy", eh,
    kwiver::vital_c::CAMERA_SPTR_CACHE.erase( cam );
  );
}


/// Create a new simple camera
vital_camera_t*
vital_camera_new( vital_eigen_matrix3x1d_t const *center,
                  vital_rotation_d_t const *rotation,
                  vital_camera_intrinsics_t const *intrinsics,
                  vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera.new", eh,
    REINTERP_TYPE( vital::vector_3d const, center, center_ptr );
    REINTERP_TYPE( vital::rotation_d const, rotation, rotation_ptr );
    vital::camera_intrinsics_sptr intrinsics_sptr
    = vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(intrinsics);
    auto c_sptr = std::make_shared<kwiver::vital::simple_camera>(
      *center_ptr, *rotation_ptr, intrinsics_sptr
    );
    vital_c::CAMERA_SPTR_CACHE.store(c_sptr);
    return reinterpret_cast< vital_camera_t* >( c_sptr.get() );
  );
  return 0;
}


/// Create a new simple camera instance with default parameters
vital_camera_t*
vital_camera_new_default( vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera.new_default", eh,
    vital::camera_sptr c_sptr = std::make_shared<vital::simple_camera>();
    vital_c::CAMERA_SPTR_CACHE.store(c_sptr);
    return reinterpret_cast< vital_camera_t* >( c_sptr.get() );
  );
  return 0;
}


/// Create a new simple camera from a string
vital_camera_t*
vital_camera_new_from_string( char const *s, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_new_from_string", eh,
    vital::camera_sptr c_sptr = std::make_shared<vital::simple_camera>();
      vital::simple_camera *sc = dynamic_cast<vital::simple_camera*>(c_sptr.get());

      std::string input_s( s );
      std::istringstream ss( input_s );
      ss >> *sc;
      vital_c::CAMERA_SPTR_CACHE.store(c_sptr);
      return reinterpret_cast< vital_camera_t* >( c_sptr.get() );
  );
  return 0;

}


/// Clone the given camera instance, returning a new camera instance
vital_camera_t*
vital_camera_clone( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_clone", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    auto c2_sptr = cam_sptr->clone();
    vital_c::CAMERA_SPTR_CACHE.store( c2_sptr );
    return reinterpret_cast< vital_camera_t* >( c2_sptr.get() );
  );
  return 0;
}


/// Get the 3D center point of the camera as a new 3x1 matrix (column-vector)
vital_eigen_matrix3x1d_t*
vital_camera_center( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_center", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    return reinterpret_cast< vital_eigen_matrix3x1d_t* >(
      new vital::vector_3d( cam_sptr->center() )
    );
  );
  return 0;
}


/// Get the 3D translation vector of the camera as a new 3x1 matrix (column-vector)
vital_eigen_matrix3x1d_t*
vital_camera_translation( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_translation", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    return reinterpret_cast< vital_eigen_matrix3x1d_t* >(
      new vital::vector_3d( cam_sptr->translation() )
    );
  );
  return 0;
}


/// Get the covariance of the camera center as a new vital covariance instance
vital_covariance_3d_t*
vital_camera_center_covar( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_center_covar", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    return reinterpret_cast< vital_covariance_3d_t* >(
      new vital::covariance_3d( cam_sptr->center_covar() )
    );
  );
  return 0;
}


/// Get rotation of the camera as a new vital rotation instance
vital_rotation_d_t*
vital_camera_rotation( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_rotation", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    return reinterpret_cast< vital_rotation_d_t* >(
      new vital::rotation_d( cam_sptr->rotation() )
    );
  );
  return 0;
}


/// Get new reference to the intrinsics of the camera
vital_camera_intrinsics_t*
vital_camera_intrinsics( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_intrinsics", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    auto i_sptr = cam_sptr->intrinsics();
    vital_c::CAMERA_INTRINSICS_SPTR_CACHE.store(i_sptr);
    return reinterpret_cast< vital_camera_intrinsics_t* >( i_sptr.get() );
  );
  return 0;
}


/// Convert camera to a 3x4 homogeneous projection matrix
vital_eigen_matrix3x4d_t*
vital_camera_as_matrix( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_as_matrix", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    return reinterpret_cast< vital_eigen_matrix3x4d_t* >(
      new vital::matrix_3x4d( cam_sptr->as_matrix() )
    );
  );
  return 0;
}


/// Project a 3D point into a (new) 2D image point via the given camera
vital_eigen_matrix2x1d_t*
vital_camera_project( vital_camera_t const *cam,
                      vital_eigen_matrix3x1d_t const *pt,
                      vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_project", eh,
    vital::camera_sptr cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    REINTERP_TYPE( vital::vector_3d const, pt, pt_ptr );
    return reinterpret_cast< vital_eigen_matrix2x1d_t* >(
      new vital::vector_2d( cam_sptr->project( *pt_ptr ) )
    );
  );
  return 0;
}


/// Compute the distance of the 3D point to the image plane
double
vital_camera_depth( vital_camera_t const *cam,
                    vital_eigen_matrix3x1d_t const *pt,
                    vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_depth", eh,
    vital::camera_sptr cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    REINTERP_TYPE( vital::vector_3d const, pt, pt_ptr );
    return cam_sptr->depth( *pt_ptr );
  );
  return 0;
}


/// Convert the camera into a string representation
char*
vital_camera_to_string( vital_camera_t const *cam, vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "vital_camera_to_string", eh,
    auto cam_sptr = vital_c::CAMERA_SPTR_CACHE.get( cam );
    std::ostringstream ss;
    ss << *cam_sptr;
    std::string ss_str( ss.str() );

    char *output = (char*)malloc( sizeof(char) * ss_str.length() );
    strcpy( output, ss_str.c_str() );
    return output;
  );
  return 0;
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
void vital_camera_write_krtd_file( vital_camera_t const *cam,
                                   char const *filepath,
                                   vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "C::camera::write_krtd_file", eh,
    kwiver::vital::camera *m_cam = kwiver::vital_c::CAMERA_SPTR_CACHE.get( cam ).get();
    kwiver::vital::write_krtd_file( *m_cam, filepath );
  );
}
