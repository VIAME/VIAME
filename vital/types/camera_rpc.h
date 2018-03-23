/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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
 * \brief Header for \link kwiver::vital::camera_perspective camera_perspective \endlink and
 *        \link kwiver::vital::camera_perspective_ camera_perspective_<T> \endlink classes
 */

#ifndef VITAL_CAMERA_RPC_H_
#define VITAL_CAMERA_RPC_H_

#include <vital/vital_export.h>

#include <iostream>
#include <memory>

#include <vital/vital_config.h>
#include <vital/types/camera.h>
#include <vital/types/vector.h>
#include <vital/logger/logger.h>


namespace kwiver {
namespace vital {

/// forward declaration of perspective camera class
class camera_rpc;
/// typedef for a camera_rpc shared pointer
typedef std::shared_ptr< camera_rpc > camera_rpc_sptr;


// ------------------------------------------------------------------
/// An abstract representation of perspective camera
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

  /// Create a clone of this camera_rpc object
  virtual camera_rpc_sptr clone() const = 0;

  // Accessors
  virtual Eigen::Matrix< double, 4, 20 > rpc_coeffs() const = 0;
  virtual vector_3d world_scale() const = 0;
  virtual vector_3d world_offset() const = 0;
  virtual vector_2d image_scale() const = 0;
  virtual vector_2d image_offset() const = 0;

  // Vector of the powers of the positions
  virtual Eigen::Matrix< double, 20, 1 > power_vector( const vector_3d& pt ) const = 0;

  /// Project a 3D point into a 2D image point
  virtual vector_2d project( const vector_3d& pt ) const;

protected:
  camera_rpc();

  kwiver::vital::logger_handle_t m_logger;

};


/// A representation of a camera
/**
 * Contains camera location, orientation, and intrinsics
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
    image_offset_(0.0, 0.0)
  {
    rpc_coeffs_ = Eigen::MatrixXd::Zero(4, 20);
    rpc_coeffs_(1, 0) = 1.0;
    rpc_coeffs_(3, 0) = 1.0;
    rpc_coeffs_(0, 1) = 1.0;
    rpc_coeffs_(2, 2) = 1.0;
  }

/// Constructor - direct from coeffs, scales, and offset
  /**
   *  This constructor constructs a camera directly from the RPC parameters
   */
  simple_camera_rpc ( vector_3d &world_scale, vector_3d &world_offset,
                      vector_2d &image_scale, vector_2d &image_offset,
                      Eigen::Matrix< double, 4, 20 > &rpc_coeffs ) :
    world_scale_( world_scale ),
    world_offset_( world_offset ),
    image_scale_( image_scale ),
    image_offset_( image_offset ),
    rpc_coeffs_( rpc_coeffs )
  { }

  /// Constructor - from base class
  simple_camera_rpc ( const camera_rpc &base ) :
    world_scale_( base.world_scale() ),
    world_offset_( base.world_offset() ),
    image_scale_( base.image_scale() ),
    image_offset_( base.image_offset() ),
    rpc_coeffs_( base.rpc_coeffs() )
  { }

  /// Create a clone of this camera object
  virtual camera_rpc_sptr clone() const
  { return camera_rpc_sptr( new simple_camera_rpc( *this ) ); }

  // Accessors
  virtual Eigen::Matrix< double, 4, 20 > rpc_coeffs() const
    { return rpc_coeffs_; }
  virtual vector_3d world_scale() const { return world_scale_; }
  virtual vector_3d world_offset() const { return world_offset_; }
  virtual vector_2d image_scale() const { return image_scale_; }
  virtual vector_2d image_offset() const { return image_offset_; }

  // Setters
  void set_rpc_coeffs(Eigen::Matrix< double, 4, 20 > coeffs)
    { rpc_coeffs_ = coeffs; }
  void set_world_scale(vector_3d& scale) { world_scale_ = scale; }
  void set_world_offset(vector_3d& offset) { world_offset_ = offset; }
  void set_image_scale(vector_2d& scale) { image_scale_ = scale; }
  void set_image_offset(vector_2d& offset) { image_offset_ = offset; }

protected:

  // Vector of the powers of the positions
  virtual Eigen::Matrix<double, 20, 1> power_vector( const vector_3d& pt ) const;

  // The RPC coefficients
  Eigen::Matrix< double, 4, 20 > rpc_coeffs_;
  // The world scale and offset
  vector_3d world_scale_;
  vector_3d world_offset_;
  // The image scale and offset
  vector_2d image_scale_;
  vector_2d image_offset_;
};


}
}   // end namespace vital


#endif // VITAL_CAMERA_RPC_H_
