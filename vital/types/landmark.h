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
 * \brief Header for \link kwiver::vital::landmark landmark \endlink objects
 */

#ifndef VITAL_LANDMARK_H_
#define VITAL_LANDMARK_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <iostream>
#include <memory>

#include "covariance.h"
#include "vector.h"
#include "color.h"

namespace kwiver {
namespace vital {


/// forward declaration of landmark class
class landmark;
/// typedef for a landmark shared pointer
typedef std::shared_ptr< landmark > landmark_sptr;

/// An abstract representation of a 3D world point.
/**
 * The base class landmark is abstract and provides a
 * double precision interface.  The templated derived class
 * can store values in either single or double precision.
 */
class landmark
{
public:
  /// Destructor
  virtual ~landmark() = default;

  /// Create a clone of this landmark object
  virtual landmark_sptr clone() const = 0;

  /// Access the type info of the underlying data (double or float)
  virtual std::type_info const& data_type() const = 0;

  /// Accessor for the world coordinates
  virtual vector_3d loc() const = 0;
  /// Accessor for the landmark scale
  virtual double scale() const = 0;
  /// Accessor for the landmark normal
  virtual vector_3d normal() const = 0;
  /// Accessor for the covariance
  virtual covariance_3d covar() const = 0;
  /// Accessor for the RGB color
  virtual rgb_color color() const = 0;
  /// Accessor for the number of observations
  virtual unsigned observations() const = 0;
  /// Accessor for the maximum intersection angle of rays triangulating this landmark
  virtual double cos_obs_angle() const = 0;

};

/// output stream operator for a base class landmark
/**
 * \param s output stream
 * \param m landmark to stream
 */
VITAL_EXPORT std::ostream& operator<<( std::ostream& s, landmark const& m );


/// A representation of a 3D world point
template < typename T >
class VITAL_EXPORT landmark_ :
  public landmark
{
public:
  /// Default Constructor
  landmark_< T >();

  /// Constructor for a landmark
  /**
   * \param loc 3D location of the landmark
   * \param scale optional scale of the landmark (default of 1)
   */
  landmark_< T >(Eigen::Matrix< T, 3, 1 > const& loc, T scale = 1);

  /// Constructor for a landmark_ from a base class landmark
  explicit landmark_< T >(landmark const& f);

  /// Create a clone of this landmark object
  virtual landmark_sptr clone() const
  {
    return landmark_sptr(new landmark_< T >(*this));
  }

  /// Access statically available type of underlying data (double or float)
  static std::type_info const& static_data_type() { return typeid(T); }

  virtual std::type_info const& data_type() const { return typeid(T); }

  /// Accessor for the world coordinates using underlying data type
  Eigen::Matrix< T, 3, 1 > const& get_loc() const { return loc_; }

  /// Accessor for the cosine of the maximum observation angle using underlying data type
  T get_cos_obs_angle() const { return cos_obs_angle_; }

  /// Accessor for the cosine of the maximum observation angle
  virtual double cos_obs_angle() const { return static_cast<double>(cos_obs_angle_); }

  /// Accessor for the world coordinates
  virtual vector_3d loc() const { return loc_.template cast< double >(); }

  /// Accessor for the landmark scale using underlying data type
  T get_scale() const { return scale_; }

  /// Accessor for the landmark scale
  virtual double scale() const { return static_cast<double> (scale_); }

  /// Accessor for the landmark normal using underlying data type
  Eigen::Matrix< T, 3, 1 > const& get_normal() const { return normal_; }

  /// Accessor for the landmark normal
  virtual vector_3d normal() const { return normal_.template cast< double >(); }

  /// Accessor for the covariance using underlying data type
  covariance_< 3, T > const& get_covar() const { return covar_; }

  /// Accessor for the covariance
  virtual covariance_3d covar() const { return static_cast<covariance_3d> (covar_); }

  /// Accessor for a const reference to the RGB color
  virtual rgb_color const& get_color() const { return color_; }

  /// Accessor for the RGB color
  virtual rgb_color color() const { return color_; }

  /// Accessor for a const reference to the number of observations
  virtual unsigned const& get_observations() const { return observations_; }

  /// Accessor for the number of observations
  virtual unsigned observations() const { return observations_; }

  /// Set the landmark position in world coordinates
  void set_loc(Eigen::Matrix< T, 3, 1 > const& loc) { loc_ = loc; }

  /// Set the scale of the landmark
  void set_scale(T scale) { scale_ = scale; }

  /// Set the landmark normal
  void set_normal(Eigen::Matrix< T, 3, 1 > const& normal) { normal_ = normal; }

  /// Set the covariance matrix of the landmark location
  void set_covar(covariance_< 3, T > const& covar) { covar_ = covar; }

  /// Set the RGB color of the landmark
  void set_color(rgb_color const& color) { color_ = color; }

  /// Set the number of observations of the landmark
  void set_observations(unsigned observations) { observations_ = observations; }

  /// Set the cosine of the maximum observation angle
  void set_cos_observation_angle(T cos_ang) { cos_obs_angle_ = cos_ang; }

protected:
  /// A vector representing the 3D position of the landmark
  Eigen::Matrix< T, 3, 1 > loc_;
  /// The scale of the landmark in 3D
  T scale_;
  /// A vector representing the normal of the landmark
  Eigen::Matrix< T, 3, 1 > normal_;
  /// Covariance representing uncertainty in the estimate of 3D position
  covariance_< 3, T > covar_;
  /// The RGB color associated with the landmark
  rgb_color color_;
  /// The number of observations that contributed to this landmark
  unsigned observations_;
  /// The cosine of the maximum intersection angle of observations that contributed to this landmark
  T cos_obs_angle_;

};


/// A double precision landmark
typedef landmark_< double > landmark_d;
/// A single precision landmark
typedef landmark_< float > landmark_f;

/// output stream operator for a landmark
template < typename T >
VITAL_EXPORT std::ostream& operator<<( std::ostream& s, landmark_< T > const& m );

/// input stream operator for a landmark
template < typename T >
VITAL_EXPORT std::istream& operator>>( std::istream& s, landmark_< T >& m );

} } // end namespace vital

#endif // VITAL_LANDMARK_H_
