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
 * \brief Interface for detected_object_set
 */

#ifndef VITAL_DETECTED_OBJECT_SET_H_
#define VITAL_DETECTED_OBJECT_SET_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/attribute_set.h>
#include <vital/noncopyable.h>

#include <vital/types/detected_object.h>

namespace kwiver {
namespace vital {

// forward declaration of detected_object class
class detected_object_set;

// typedef for a detected_object shared pointer
typedef std::shared_ptr< detected_object_set > detected_object_set_sptr;

// ----------------------------------------------------------------
/**
 * @brief Set of detected objects.
 *
 * This class represents a ordered set of detected objects. The
 * detections are ordered on their basic confidence value.
 *
 * Reentrancy considerations: Typical usage for a set is for a single
 * detector thread to create a set. It is possible to have an
 * application where two threads are accessing the same set
 * concurrently.
 */
class VITAL_EXPORT detected_object_set
  : private noncopyable
{
public:

  /**
   * @brief Create an empty detection set.
   *
   * This CTOR creates an empty detection set. Detections can be added
   * with the add() method.
   */
  detected_object_set();

  ~detected_object_set() VITAL_DEFAULT_DTOR

  /**
   * @brief Create new set of detected objects.
   *
   * This CTOR creates a detection set using the supplied vector of
   * detection objects. This can be used to create a new detection set
   * from the output of a select() method.
   *
   * @param objs Vector of detected objects.
   */
  detected_object_set( detected_object::vector_t const& objs );

  /**
   * @brief Create deep copy.
   *
   * This method creates a deep copy of this object.
   *
   * @return Managed copy of this object.
   */
  detected_object_set_sptr clone () const;

  /**
   * @brief Add detection to set.
   *
   * This method adds a new detection to this set.
   *
   * @param object Detection to be added to set.
   */
  void add( detected_object_sptr object );

  /**
   * @brief Get number of detections in this set.
   *
   * This method returns the number of detections in the set.
   *
   * @return Number of detections.
   */
  size_t size() const;

  /**
   * @brief Select detections based on confidence value.
   *
   * This method returns a vector of detections ordered by confidence
   * value, high to low. If the optional threshold is specified, then
   * all detections from the set that are less than the threshold are
   * not in the selected set. Note that the selected set may be empty.
   *
   * The returned vector refers to the actual detections in the set,
   * so if you make changes to the selected set, you are also changing
   * the object in the set. If you want a clean set of detections,
   * call clone() first.
   *
   * @param threshold Select all detections with confidence not less
   *                  than this value. If this parameter is omitted,
   *                  then all detections are selected.
   *
   * @return List of detections.
   */
  detected_object::vector_t select( double threshold = detected_object_type::INVALID_SCORE ) const;

  /**
   * @brief Select detections based on class_name
   *
   * This method returns a vector of detections that have the
   * specified class_name. These detections are ordered by descending
   * score for the name. Note that the selected set may be empty.
   *
   * The returned vector refers to the actual detections in the set,
   * so if you make changes to the selected set, you are also changing
   * the object in the set. If you want a clean set of detections,
   * call clone() first.
   *
   * @param class_name class name
   * @param threshold Select all detections with confidence not less
   *                  than this value. If this parameter is omitted,
   *                  then all detections with the label are selected.
   *
   * @return List of detections.
   */
  detected_object::vector_t select( const std::string& class_name,
                                    double             threshold = detected_object_type::INVALID_SCORE ) const;

  /**
   * @brief Get attributes set.
   *
   * This method returns a pointer to the attribute set that is
   * attached to this object. It is possible that the pointer is NULL,
   * so check before using it.
   *
   * @return Pointer to attribute set or NULL
   */
  attribute_set_sptr attributes() const;

  /**
   * @brief Attach attributes set to this object.
   *
   * This method attaches the specified attribute set to this object.
   *
   * @param attrs Pointer to attribute set to attach.
   */
  void set_attributes( attribute_set_sptr attrs );

private:
  // List of detections ordered by confidence value.
  detected_object::vector_t m_detected_objects;

  attribute_set_sptr m_attrs;
};

} } // end namespace

#endif
