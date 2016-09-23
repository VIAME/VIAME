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
 * \brief Interface for detected_object class
 */

#ifndef VITAL_DETECTED_OBJECT_H_
#define VITAL_DETECTED_OBJECT_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>

#include <memory>
#include <vector>

#include <vital/types/detected_object_type.h>
#include <vital/types/vector.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>

#include <vital/io/eigen_io.h>
#include <Eigen/Geometry>

namespace kwiver {
namespace vital {

// forward declaration of detected_object class
class detected_object;

// typedef for a detected_object shared pointer
typedef std::shared_ptr< detected_object > detected_object_sptr;


// ----------------------------------------------------------------
/**
 * @brief Detected object class.
 *
 * This class represents a detected object in image space.
 *
 * There is one object of this type for each detected object. These
 * objects are defined by a bounding box in the image space. Each
 * object has an optional classification object attached.
 *
 */
class VITAL_EXPORT detected_object
{
public:
  typedef std::vector< detected_object_sptr > vector_t;


  /**
   * @brief Create an empty object
   */
  detected_object();

  /**
   * @brief Create detected object with bounding box and other attributes.
   *
   * @param bbox Bounding box surrounding detected object, in image coordinates.
   * @param confidence Detectors confidence in this detection.
   * @param classifications Optional object classification.
   */
  detected_object( const bounding_box_d& bbox,
                   double confidence = 1.0,
                   detected_object_type_sptr classifications = detected_object_type_sptr() );

  virtual ~detected_object() VITAL_DEFAULT_DTOR

  /**
   * @brief Get bounding box from this detection.
   *
   * The bounding box for this detection is returned. This box is in
   * image coordinates. A default constructed (invalid) bounding box
   * is returned if no box has been supplied for this detection.
   *
   * @return A copy of the bounding box.
   */
  bounding_box_d bounding_box() const;

  /**
   * @brief Set new bounding box for this detection.
   *
   * The supplied bounding box replaces the box for this detection.
   *
   * @param bbox Bounding box for this detection.
   */
  void set_bounding_box( const bounding_box_d& bbox );

  /**
   * @brief Get confidence for this detection.
   *
   * This method returns the current confidence value for this detection.
   * Confidence values are in the range of 0.0 - 1.0.
   *
   * @return Confidence value for this detection.
   */
  double confidence() const;

  /**
   * @brief Set new confidence value for detection.
   *
   * This method sets a new confidence value for this detection.
   * Confidence values are in the range of [0.0 - 1.0].
   *
   * @param d New confidence value for this detection.
   */
  void set_confidence( double d );

  /**
   * @brief Get pointer to optional classifications object.
   *
   * This method returns the pointer to the classification object if
   * there is one. If there is no classification object the pointer is
   * NULL.
   *
   * @return Pointer to classification object or NULL.
   */
  detected_object_type_sptr type();

  /**
   * @brief Set new classifications for this detection.
   *
   * This method supplies a new set of class_names and scores for this
   * detection.
   *
   * @param c New classification for this detection
   */
  void set_type( detected_object_type_sptr c );

  /**
   * @brief Get pointer to optional classifications object.
   *
   * This method returns the pointer to the classification object if
   * there is one. If there is no classification object the pointer is
   * NULL.
   *
   * @return Pointer to classification object or NULL.
   */
  image_container_sptr mask();

  /**
   * @brief Set new classifications for this detection.
   *
   * This method supplies a new set of class_names and scores for this
   * detection.
   *
   * @param c New classification for this detection
   */
  void set_mask( image_container_sptr m );

private:
  std::shared_ptr< bounding_box_d > m_bounding_box;
  double m_confidence;
  image_container_sptr m_image;

  // The detection type is an optional list of possible object types.
  detected_object_type_sptr m_type;
};

} }

#endif
