// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for \link kwiver::vital::camera camera \endlink and
 *        \link kwiver::vital::camera_ camera_<T> \endlink classes
 */

#ifndef VITAL_CAMERA_H_
#define VITAL_CAMERA_H_

#include <vital/vital_export.h>

#include <iostream>
#include <memory>
#include <vector>

#include <vital/types/vector.h>

namespace kwiver {
namespace vital {

/// forward declaration of camera class
class camera;
/// typedef for a camera shared pointer
typedef std::shared_ptr< camera > camera_sptr;
/// typedef for a vector of cameras
typedef std::vector< camera_sptr > camera_sptr_list;

// ------------------------------------------------------------------
/// An abstract representation of camera
/**
 * The base class of cameras.
 */
class VITAL_EXPORT camera
{
public:
  /// Destructor
  virtual ~camera() = default;

  /// Create a clone of this camera object
  virtual camera_sptr clone() const = 0;

  /// Project a 3D point into a 2D image point
  virtual vector_2d project( const vector_3d& pt ) const = 0;

  /// Accessor for the image width
  virtual unsigned int image_width() const = 0;

  /// Accessor for the image height
  virtual unsigned int image_height() const = 0;

protected:
  camera() {};
};

}
}   // end namespace vital

#endif // VITAL_CAMERA_H_
