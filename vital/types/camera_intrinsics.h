/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief Header for \link kwiver::vital::camera_intrinsics camera_intrinsics
 *        \endlink class
 */

#ifndef VITAL_CAMERA_INTRINSICS_H_
#define VITAL_CAMERA_INTRINSICS_H_

#include <vital/vital_export.h>

#include <iostream>
#include <vector>
#include <memory>

#include <vital/types/matrix.h>
#include <vital/types/vector.h>


namespace kwiver {
namespace vital {

/// forward declaration of camera intrinsics class
class camera_intrinsics;
/// typedef for a camera intrinsics shared pointer
typedef std::shared_ptr< camera_intrinsics > camera_intrinsics_sptr;


// ------------------------------------------------------------------
/// An abstract representation of camera intrinsics
class VITAL_EXPORT camera_intrinsics
{
public:
  /// Destructor
  virtual ~camera_intrinsics() { }

  /// Create a clone of this object
  virtual camera_intrinsics_sptr clone() const = 0;

  /// Access the focal length
  virtual double focal_length() const = 0;
  /// Access the principal point
  virtual vector_2d principal_point() const = 0;
  /// Access the aspect ratio
  virtual double aspect_ratio() const = 0;
  /// Access the skew
  virtual double skew() const = 0;
  /// Access the distortion coefficients
  virtual std::vector<double> dist_coeffs() const
  { return std::vector<double>(); }

  /// Access the intrinsics as an upper triangular matrix
  /**
   *  \note This matrix includes the focal length, principal point,
   *  aspect ratio, and skew, but does not model distortion
   */
  virtual matrix_3x3d as_matrix() const;

  /// Map normalized image coordinates into actual image coordinates
  /**
   *  This function applies both distortion and application of the
   *  calibration matrix to map into actual image coordinates
   */
  virtual vector_2d map(const vector_2d& norm_pt) const;

  /// Map a 3D point in camera coordinates into actual image coordinates
  virtual vector_2d map(const vector_3d& norm_hpt) const;

  /// Unmap actual image coordinates back into normalized image coordinates
  /**
   *  This function applies both application of the inverse calibration matrix
   *  and undistortion of the normalized coordinates.
   */
  virtual vector_2d unmap(const vector_2d& norm_pt) const;

  /// Map normalized image coordinates into distorted coordinates
  /**
   *  The default implementation is the identity transformation (no distortion)
   */
  virtual vector_2d distort(const vector_2d& norm_pt) const { return norm_pt; };

  /// Unmap distorted normalized coordinates into normalized coordinates
  /**
   *  The default implementation is the identity transformation (no distortion)
   */
  virtual vector_2d undistort(const vector_2d& dist_pt) const { return dist_pt; };

};

/// output stream operator for a base class camera_intrinsics
VITAL_EXPORT
std::ostream& operator<<( std::ostream& s, const camera_intrinsics& c );



/// A representation of camera intrinsic parameters
class VITAL_EXPORT simple_camera_intrinsics :
  public camera_intrinsics
{
public:
  /// typedef for Eigen dynamic vector
  typedef Eigen::VectorXd vector_t;

  /// Default Constructor
  simple_camera_intrinsics()
  : focal_length_(1.0),
    principal_point_(0.0, 0.0),
    aspect_ratio_(1.0),
    skew_(0.0),
    dist_coeffs_()
  {}

  /// Constructor for camera intrinsics
  simple_camera_intrinsics(const double focal_length,
                           const vector_2d& principal_point,
                           const double aspect_ratio=1.0,
                           const double skew=0.0,
                           const vector_t dist_coeffs=vector_t())
  : focal_length_(focal_length),
    principal_point_(principal_point),
    aspect_ratio_(aspect_ratio),
    skew_(skew),
    dist_coeffs_(dist_coeffs)
  {}

  /// Constructor from the base class
  explicit simple_camera_intrinsics(const camera_intrinsics& base)
  : focal_length_(base.focal_length()),
    principal_point_(base.principal_point()),
    aspect_ratio_(base.aspect_ratio()),
    skew_(base.skew())
  {
    std::vector<double> dc = base.dist_coeffs();
    dist_coeffs_ = vector_t::Map(dc.data(), dc.size());
  }

  /// Create a clone of this object
  virtual camera_intrinsics_sptr clone() const
  { return camera_intrinsics_sptr( new simple_camera_intrinsics( *this ) ); }

  /// Constructor - from a calibration matrix
  /**
   * \note ignores values below the diagonal
   * \param K calibration matrix to construct from
   */
  explicit simple_camera_intrinsics(const matrix_3x3d& K,
                                    const vector_t& d=vector_t());

  /// Access the focal length
  virtual double focal_length() const { return focal_length_; }
  /// Access the principal point
  virtual vector_2d principal_point() const { return principal_point_; }
  /// Access the aspect ratio
  virtual double aspect_ratio() const { return aspect_ratio_; }
  /// Access the skew
  virtual double skew() const { return skew_; }
  /// Access the distortion coefficients
  virtual std::vector<double> dist_coeffs() const
  {
    return std::vector<double>(dist_coeffs_.data(), dist_coeffs_.data() + dist_coeffs_.size());
  }

  /// Access the focal length
  const double& get_focal_length() const { return focal_length_; }
  /// Access the principal point
  const vector_2d& get_principal_point() const { return principal_point_; }
  /// Access the aspect ratio
  const double& get_aspect_ratio() const { return aspect_ratio_; }
  /// Access the skew
  const double& get_skew() const { return skew_; }
  /// Access the distortion coefficients
  const vector_t& get_dist_coeffs() const { return dist_coeffs_; }

  /// Set the focal length
  void set_focal_length(const double& focal_length) { focal_length_ = focal_length; }
  /// Set the principal point
  void set_principal_point(const vector_2d& pp) { principal_point_ = pp; }
  /// Set the aspect_ratio
  void set_aspect_ratio(const double& aspect_ratio) { aspect_ratio_ = aspect_ratio; }
  /// Set the skew
  void set_skew(const double& skew) { skew_ = skew; }
  /// Set the distortion coefficients
  void set_dist_coeffs(const vector_t& d) { dist_coeffs_ = d; }

  /// Map normalized image coordinates into distorted coordinates
  virtual vector_2d distort(const vector_2d& norm_pt) const;

  /// Unnap distorted normalized coordinates into normalized coordinates
  /** \note applying inverse distortion is not closed form, so this function
   *  uses an iterative solver.
   */
  virtual vector_2d undistort(const vector_2d& dist_pt) const;

protected:
  /// focal length of camera
  double focal_length_;
  /// principal point of camera
  vector_2d principal_point_;
  /// aspect ratio of camera
  double aspect_ratio_;
  /// skew of camera
  double skew_;
  /// Lens distortion coefficients
  vector_t dist_coeffs_;
};


/// input stream operator for camera intrinsics
/**
 * \param s input stream
 * \param k simple_camera_intrinsics to stream into
 */
VITAL_EXPORT std::istream&
operator>>(std::istream& s, simple_camera_intrinsics& k);


} } // end namespace

#endif // VITAL_CAMERA_INTRINSICS_H_
