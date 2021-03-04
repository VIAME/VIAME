// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
  /// Access the image width
  virtual unsigned int image_width() const = 0;
  /// Access the image height
  virtual unsigned int image_height() const = 0;
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

  /// Check if a normalized image coordinate can map into image coordinates
  /**
  *  Some points may lie outside the domain of the mapping function and produce
  *  invalid results.  This function tests if the point lies in the valid domain
  */
  virtual bool is_map_valid(const vector_2d& norm_pt) const { return true; }

  /// Check if a 3D point in camera coordinates can map into image coordinates
  /**
  *  Some points may lie outside the domain of the mapping function and produce
  *  invalid results.  This function tests if the point lies in the valid domain
  */
  virtual bool is_map_valid(const vector_3d& norm_hpt) const;
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
    dist_coeffs_(),
    image_width_(0),
    image_height_(0),
    max_distort_radius_sq_(std::numeric_limits<double>::infinity())
  {}

  /// Constructor for camera intrinsics
  simple_camera_intrinsics(const double focal_length,
                           const vector_2d& principal_point,
                           const double aspect_ratio=1.0,
                           const double skew=0.0,
                           const vector_t dist_coeffs=vector_t(),
                           const unsigned int image_width=0,
                           const unsigned int image_height=0)
  : focal_length_(focal_length),
    principal_point_(principal_point),
    aspect_ratio_(aspect_ratio),
    skew_(skew),
    dist_coeffs_(dist_coeffs),
    image_width_(image_width),
    image_height_(image_height),
    max_distort_radius_sq_(compute_max_distort_radius_sq())
  {}

  /// Constructor from the base class
  explicit simple_camera_intrinsics(const camera_intrinsics& base)
  : focal_length_(base.focal_length()),
    principal_point_(base.principal_point()),
    aspect_ratio_(base.aspect_ratio()),
    skew_(base.skew()),
    image_width_(base.image_width()),
    image_height_(base.image_height()),
    max_distort_radius_sq_(std::numeric_limits<double>::infinity())
  {
    std::vector<double> dc = base.dist_coeffs();
    dist_coeffs_ = vector_t::Map(dc.data(), dc.size());
    max_distort_radius_sq_ = compute_max_distort_radius_sq();
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
  /// Access the image width
  virtual unsigned int image_width() const { return image_width_; }
  /// Access the image height
  virtual unsigned int image_height() const { return image_height_; }
  /// Access the distortion coefficients
  virtual std::vector<double> dist_coeffs() const
  {
    return std::vector<double>(dist_coeffs_.data(), dist_coeffs_.data() + dist_coeffs_.size());
  }
  /// Access the maximum distortion radius
  double max_distort_radius() const { return std::sqrt(max_distort_radius_sq_); }

  /// Access the focal length
  const double& get_focal_length() const { return focal_length_; }
  /// Access the principal point
  const vector_2d& get_principal_point() const { return principal_point_; }
  /// Access the aspect ratio
  const double& get_aspect_ratio() const { return aspect_ratio_; }
  /// Access the skew
  const double& get_skew() const { return skew_; }
  /// Access the image width
  const unsigned int& get_image_width() const { return image_width_; }
  /// Access the image height
  const unsigned int& get_image_height() const { return image_height_; }
  /// Access the distortion coefficients
  const vector_t& get_dist_coeffs() const { return dist_coeffs_; }
  /// Access the maximum distortion radius squared
  const double& get_max_distort_radius_sq() const { return max_distort_radius_sq_; }

  /// Set the focal length
  void set_focal_length(const double& focal_length) { focal_length_ = focal_length; }
  /// Set the principal point
  void set_principal_point(const vector_2d& pp) { principal_point_ = pp; }
  /// Set the aspect_ratio
  void set_aspect_ratio(const double& aspect_ratio) { aspect_ratio_ = aspect_ratio; }
  /// Set the skew
  void set_skew(const double& skew) { skew_ = skew; }
  /// Set the image width
  void set_image_width(const unsigned int width) { image_width_ = width; }
  /// Set the image height
  void set_image_height(const unsigned int height) { image_height_ = height; }
  /// Set the distortion coefficients
  void set_dist_coeffs(const vector_t& d)
  {
    dist_coeffs_ = d;
    max_distort_radius_sq_ = compute_max_distort_radius_sq();
  }

  /// Map normalized image coordinates into distorted coordinates
  virtual vector_2d distort(const vector_2d& norm_pt) const;

  /// Unnap distorted normalized coordinates into normalized coordinates
  /** \note applying inverse distortion is not closed form, so this function
   *  uses an iterative solver.
   */
  virtual vector_2d undistort(const vector_2d& dist_pt) const;

  /// Check if a normalized image coordinate can map into image coordinates
  /**
  *  Tests if a point lies beyond the maximum distortion radius
  */
  virtual bool is_map_valid(const vector_2d& norm_pt) const;

  using camera_intrinsics::is_map_valid;

  /// Compute maximum squared radius for radial distortion given coefficients
  /** A point at radius r is distorted to \f$(1 + a r^2 + b r^4 + c r^6) r\f$.
   *  This function computes the maximum value of r before the function
   *  curves back on itself (i.e. the slope is negative).  In many cases
   *  the maximum radius is infinity.  Beyond this maximum radius the
   *  distortion function is no longer injective.  Points beyond this radius
   *  can project into the image bounds even if they are far outside the
   *  field of view.
   */
  static double max_distort_radius_sq(double a, double b, double c);

protected:
  /// Compute the maximum distortion radius (squared) from dist_coeffs_
  virtual double compute_max_distort_radius_sq() const;

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
  /// Image width
  unsigned int image_width_;
  /// Image height
  unsigned int image_height_;
  /// maximum distortion radius (squared)
  /** Do not trust the radial distortion of points beyond this radius.
   *  The value is stored as radius squared because we mostly work
   *  with squared values for efficiency (avoids many square roots)
   */
  double max_distort_radius_sq_;
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
