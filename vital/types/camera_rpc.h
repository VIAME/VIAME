// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for \link kwiver::vital::camera_rpc camera_rpc \endlink and
 *        \link kwiver::vital::camera_rpc_ camera_rpc_<T> \endlink classes
 */

#ifndef VITAL_CAMERA_RPC_H_
#define VITAL_CAMERA_RPC_H_

#include <vital/vital_export.h>

#include <iostream>
#include <memory>

#include <vital/vital_config.h>
#include <vital/types/camera.h>
#include <vital/types/matrix.h>
#include <vital/types/vector.h>
#include <vital/logger/logger.h>

namespace kwiver {
namespace vital {

typedef Eigen::Matrix< double, 4, 20 > rpc_matrix;
typedef Eigen::Matrix< double, 4, 10 > rpc_deriv_matrix;

/// forward declaration of rpc camera class
class camera_rpc;
/// typedef for a camera_rpc shared pointer
typedef std::shared_ptr< camera_rpc > camera_rpc_sptr;

// ------------------------------------------------------------------
/// An abstract representation of rpc camera
/**
 * The base class of camera_rpcs is abstract and provides a
 * double precision interface.  The templated derived class
 * can store values in either single or double precision.
 */
class VITAL_EXPORT camera_rpc : public camera
{
public:
  /// Destructor
  virtual ~camera_rpc() = default;

  /// Vector of the powers of the 3D positions
  /**
   * Produces a vector whose components are the various powers (up to cubic)
   * of the components of a 3D position. This vector is used to construct
   * the various polynomials used in the RPC model. The order of the terms is
   *
   * {1, x, y, z, xy, xz, yz, xx, yy, zz, xyz, xxx, xyy, xzz, xxy, yyy, yzz, xxz, yyz, zzz}
   */
  static Eigen::Matrix<double, 20, 1> power_vector( const vector_3d& pt );

  /// Create a clone of this camera_rpc object
  virtual camera_sptr clone() const = 0;

  // Accessors
  virtual rpc_matrix rpc_coeffs() const = 0;
  virtual vector_3d world_scale() const = 0;
  virtual vector_3d world_offset() const = 0;
  virtual vector_2d image_scale() const = 0;
  virtual vector_2d image_offset() const = 0;
  virtual unsigned int image_width() const = 0;
  virtual unsigned int image_height() const = 0;

  /// Project a 3D point into a 2D image point
  virtual vector_2d project( const vector_3d& pt ) const;

  /// Project a 2D image back to a 3D point in space
  virtual vector_3d back_project( const vector_2d& image_pt, double elev ) const;

protected:
  camera_rpc();

  // Compute the Jacobian of the RPC at the given normalized world point
  // Currently this only computes the 2x2 Jacobian for X and Y parameters.
  // This function also returns the normalized projected point
  virtual void jacobian( const vector_3d& pt, matrix_2x2d& J, vector_2d& norm_pt ) const = 0;

  kwiver::vital::logger_handle_t m_logger;

};

/// A representation of a camera
/**
 * Contains camera rpc coefficients, offsets, and scales
 */
class VITAL_EXPORT simple_camera_rpc :
  public camera_rpc
{
public:
  /// Default Constructor - creates identity camera
  simple_camera_rpc ( ) :
    world_scale_(1.0, 1.0, 1.0),
    world_offset_(0.0, 0.0, 0.0),
    image_scale_(1.0, 1.0),
    image_offset_(0.0, 0.0),
    image_width_(0),
    image_height_(0)
  {
    rpc_coeffs_ = rpc_matrix::Zero();
    rpc_coeffs_(1, 0) = 1.0;
    rpc_coeffs_(3, 0) = 1.0;
    rpc_coeffs_(0, 1) = 1.0;
    rpc_coeffs_(2, 2) = 1.0;

    update_partial_deriv();
  }

  /// Constructor - direct from coeffs, scales, and offset
  /**
   *  This constructor constructs a camera directly from the RPC parameters
   */
  simple_camera_rpc ( vector_3d &world_scale, vector_3d &world_offset,
                      vector_2d &image_scale, vector_2d &image_offset,
                      rpc_matrix &rpc_coeffs, unsigned int image_width=0,
                      unsigned int image_height=0) :
    world_scale_( world_scale ),
    world_offset_( world_offset ),
    image_scale_( image_scale ),
    image_offset_( image_offset ),
    rpc_coeffs_( rpc_coeffs ),
    image_width_( image_width ),
    image_height_( image_height )
  {
    update_partial_deriv();
  }

  /// Constructor - from base class
  simple_camera_rpc ( const camera_rpc &base ) :
    world_scale_( base.world_scale() ),
    world_offset_( base.world_offset() ),
    image_scale_( base.image_scale() ),
    image_offset_( base.image_offset() ),
    rpc_coeffs_( base.rpc_coeffs() ),
    image_width_( base.image_width() ),
    image_height_( base.image_height() )
  {
    update_partial_deriv();
  }

  /// Create a clone of this camera object
  virtual camera_sptr clone() const
  { return camera_sptr( std::make_shared< simple_camera_rpc >( *this ) ); }

  // Accessors
  virtual rpc_matrix rpc_coeffs() const
    { return rpc_coeffs_; }
  virtual vector_3d world_scale() const { return world_scale_; }
  virtual vector_3d world_offset() const { return world_offset_; }
  virtual vector_2d image_scale() const { return image_scale_; }
  virtual vector_2d image_offset() const { return image_offset_; }
  virtual unsigned int image_width() const { return image_width_; }
  virtual unsigned int image_height() const { return image_height_; }

  // Setters
  void set_rpc_coeffs(rpc_matrix coeffs)
  {
    rpc_coeffs_ = coeffs;
    update_partial_deriv();
  }
  void set_world_scale(vector_3d const& scale) { world_scale_ = scale; }
  void set_world_offset(vector_3d const& offset) { world_offset_ = offset; }
  void set_image_scale(vector_2d const& scale) { image_scale_ = scale; }
  void set_image_offset(vector_2d const& offset) { image_offset_ = offset; }
  void set_image_width(unsigned int width) { image_width_ = width; }
  void set_image_height(unsigned int height) { image_height_ = height; }

protected:

  virtual void jacobian( const vector_3d& pt, matrix_2x2d& J, vector_2d& norm_pt ) const;

  // Update the partial derivatives needed to compute the jacobian
  void update_partial_deriv() const;

  // The RPC coefficients
  rpc_matrix rpc_coeffs_;
  // The partial derivatives coefficients
  mutable rpc_deriv_matrix dx_coeffs_;
  mutable rpc_deriv_matrix dy_coeffs_;
  // The world scale and offset
  vector_3d world_scale_;
  vector_3d world_offset_;
  // The image scale and offset
  vector_2d image_scale_;
  vector_2d image_offset_;
  // The image width and height
  unsigned int image_width_;
  unsigned int image_height_;
};

}
}   // end namespace vital

#endif // VITAL_CAMERA_RPC_H_
