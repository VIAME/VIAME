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
