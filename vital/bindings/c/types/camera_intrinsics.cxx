// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
