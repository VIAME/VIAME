/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief C Interface implementation to \p vital::camera_intrinsics class
 */

#include "camera_intrinsics.h"

#include <memory>

#include <vital/bindings/c/helpers/camera_intrinsics.h>
#include <vital/types/camera_intrinsics.h>


namespace kwiver {
namespace vital_c {

SharedPointerCache< vital::camera_intrinsics, vital_camera_intrinsics_t >
  CAMERA_INTRINSICS_SPTR_CACHE( "camera_intrinsics" );

} // end namespace vital_c
} // end namespace kwiver


/// Create new simple camera intrinsics object with default parameters
vital_camera_intrinsics_t*
vital_camera_intrinsics_new_default( vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.new_default", eh,

    auto ci_sptr = std::make_shared<kwiver::vital::simple_camera_intrinsics>();
    kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.store( ci_sptr );
    return reinterpret_cast<vital_camera_intrinsics_t*>( ci_sptr.get() );

  );
  return 0;
}


/**
 * Create a new simple camera intrinsics object with specified focal length and
 * principle point.
 */
vital_camera_intrinsics_t*
vital_camera_intrinsics_new_partial( double focal_length,
                                     vital_eigen_matrix2x1d_t *principle_point,
                                     vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.new_partial", eh,

    kwiver::vital::vector_2d *pp
      = reinterpret_cast<kwiver::vital::vector_2d*>(principle_point);
    auto ci_sptr = std::make_shared<kwiver::vital::simple_camera_intrinsics>(
      focal_length, *pp
    );
    kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.store( ci_sptr );
    return reinterpret_cast<vital_camera_intrinsics_t*>( ci_sptr.get() );

  );
  return 0;
}


/// Create a new simple camera intrinsics object
vital_camera_intrinsics_t*
vital_camera_intrinsics_new( double focal_length,
                             vital_eigen_matrix2x1d_t *principle_point,
                             double aspect_ratio,
                             double skew,
                             vital_eigen_matrixXx1d_t *dist_coeffs,
                             vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.new", eh,

    kwiver::vital::vector_2d *pp
      = reinterpret_cast<kwiver::vital::vector_2d*>(principle_point);

    // Build up temporary VectorXd for constructor
    Eigen::VectorXd *coeff_vec
      = reinterpret_cast<Eigen::VectorXd*>( dist_coeffs );

    auto ci_sptr = std::make_shared<kwiver::vital::simple_camera_intrinsics>(
      focal_length, *pp, aspect_ratio, skew, *coeff_vec
    );
    kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.store( ci_sptr );
    return reinterpret_cast<vital_camera_intrinsics_t*>( ci_sptr.get() );
  );
  return 0;
}


/// Destroy a given non-null camera intrinsics object
void
vital_camera_intrinsics_destroy( vital_camera_intrinsics_t *ci,
                                 vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.destroy", eh,
    kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.erase( ci );
  );
}


/// Get the focal length
double
vital_camera_intrinsics_get_focal_length( vital_camera_intrinsics_t *ci,
                                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.get_focal_length", eh,
    return kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get( ci )->focal_length();
  );
  return 0;
}


/// Get a new copy of the principle point
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_get_principle_point( vital_camera_intrinsics_t *ci,
                                             vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.get_principle_point", eh,
    Eigen::Vector2d *w = new Eigen::Vector2d(
      kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->principal_point()
    );
    return reinterpret_cast< vital_eigen_matrix2x1d_t* >( w );
  );
  return 0;
}


/// Get the aspect ratio
double
vital_camera_intrinsics_get_aspect_ratio( vital_camera_intrinsics_t *ci,
                                          vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.get_aspect_ratio", eh,
    return kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->aspect_ratio();
  );
  return 0;
}


/// Get the skew value
double
vital_camera_intrinsics_get_skew( vital_camera_intrinsics_t *ci,
                                  vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.get_skew", eh,
    return kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->skew();
  );
  return 0;
}


/// Get the distance coefficients
vital_eigen_matrixXx1d_t*
vital_camera_intrinsics_get_dist_coeffs( vital_camera_intrinsics_t *ci,
                                         vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.get_dist_coeffs", eh,
    std::vector<double> v =
      kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->dist_coeffs();
    Eigen::VectorXd *e = new Eigen::VectorXd(
        Eigen::VectorXd::Map( v.data(), v.size() )
    );
    return reinterpret_cast< vital_eigen_matrixXx1d_t* >( e );
  );
  return 0;
}


/// Access the intrinsics as an upper triangular matrix
vital_eigen_matrix3x3d_t*
vital_camera_intrinsics_as_matrix( vital_camera_intrinsics_t *ci,
                                   vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.as_matrix", eh,
    kwiver::vital::matrix_3x3d *m = new kwiver::vital::matrix_3x3d(
      kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->as_matrix()
    );
    return reinterpret_cast< vital_eigen_matrix3x3d_t* >( m );
  );
  return 0;
}


/// Map normalized image coordinates into actual image coordinates
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_map_2d( vital_camera_intrinsics_t *ci,
                                vital_eigen_matrix2x1d_t *p,
                                vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.map_2d", eh,
    kwiver::vital::vector_2d &p2 = *reinterpret_cast<kwiver::vital::vector_2d*>(p);
    kwiver::vital::vector_2d *v = new kwiver::vital::vector_2d(
      kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->map(p2)
    );
    return reinterpret_cast<vital_eigen_matrix2x1d_t*>( v );
  );
  return 0;
}


/// Map a 3D point in camera coordinates into actual image coordinates
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_map_3d( vital_camera_intrinsics_t *ci,
                                vital_eigen_matrix3x1d_t *p,
                                vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.map_3d", eh,
    kwiver::vital::vector_3d &p2 = *reinterpret_cast<kwiver::vital::vector_3d*>(p);
    kwiver::vital::vector_2d *v = new kwiver::vital::vector_2d(
      kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->map(p2)
    );
    return reinterpret_cast<vital_eigen_matrix2x1d_t*>( v );
  );
  return 0;
}


/// Unmap actual image coordinates back into normalized image coordinates
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_unmap_2d( vital_camera_intrinsics_t *ci,
                                  vital_eigen_matrix2x1d_t *p,
                                  vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.unmap_2d", eh,
    kwiver::vital::vector_2d &p2 = *reinterpret_cast<kwiver::vital::vector_2d*>(p);
      kwiver::vital::vector_2d *v = new kwiver::vital::vector_2d(
        kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->unmap(p2)
      );
      return reinterpret_cast<vital_eigen_matrix2x1d_t*>( v );
  );
  return 0;
}


/// Map normalized image coordinates into distorted coordinates
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_distort_2d( vital_camera_intrinsics_t *ci,
                                    vital_eigen_matrix2x1d_t *p,
                                    vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.distort_2d", eh,
    kwiver::vital::vector_2d &p2 = *reinterpret_cast<kwiver::vital::vector_2d*>(p);
      kwiver::vital::vector_2d *v = new kwiver::vital::vector_2d(
        kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->distort(p2)
      );
      return reinterpret_cast<vital_eigen_matrix2x1d_t*>( v );
  );
  return 0;
}


/// Unmap distorted normalized coordinates into normalized coordinates
vital_eigen_matrix2x1d_t*
vital_camera_intrinsics_undistort_2d( vital_camera_intrinsics_t *ci,
                                      vital_eigen_matrix2x1d_t *p,
                                      vital_error_handle_t *eh )
{
  STANDARD_CATCH(
    "camera_intrinsics.undistort_2d", eh,
    kwiver::vital::vector_2d &p2 = *reinterpret_cast<kwiver::vital::vector_2d*>(p);
      kwiver::vital::vector_2d *v = new kwiver::vital::vector_2d(
        kwiver::vital_c::CAMERA_INTRINSICS_SPTR_CACHE.get(ci)->undistort(p2)
      );
      return reinterpret_cast<vital_eigen_matrix2x1d_t*>( v );
  );
  return 0;
}
