/*ckwg +29
 * Copyright 2016-2017,2019 by Kitware, Inc.
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
#include <vital/set.h>

#include <vital/types/detected_object.h>

namespace kwiver {
namespace vital {

// forward declaration of detected_object class
class detected_object_set;

// typedef for a detected_object shared pointer
using detected_object_set_sptr = std::shared_ptr< detected_object_set >;
using detected_object_set_scptr = std::shared_ptr< detected_object_set const >;

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
  : public set< detected_object_sptr >
  , private noncopyable
{
public:

  /**
   * @brief Create an empty detection set.
   *
   * This CTOR creates an empty detection set. Detections can be added
   * with the add() method.
   */
  detected_object_set();

  ~detected_object_set() = default;

  /**
   * @brief Create new set of detected objects.
   *
   * This CTOR creates a detection set using the supplied vector of
   * detection objects. This can be used to create a new detection set
   * from the output of a select() method.
   *
   * @param objs Vector of detected objects.
   */
  detected_object_set( std::vector< detected_object_sptr > const& objs );

  /**
   * @brief Create deep copy.
   *
   * This method creates a deep copy of this object.
   *
   * @return Managed copy of this object.
   */
  detected_object_set_sptr clone() const;

  /**
   * @brief Add detection to set.
   *
   * This method adds a new detection to this set.
   *
   * @param object Detection to be added to set.
   */
  void add( detected_object_sptr object );

  /**
   * @brief Add detection set to set.
   *
   * This method adds a new detection set to this set.
   *
   * @param detections Detection set to be added to set.
   */
  void add( detected_object_set_sptr detections );

  /**
   * @brief Get number of detections in this set.
   *
   * This method returns the number of detections in the set.
   *
   * @return Number of detections.
   */
  size_t size() const override;

  /**
   * @brief Returns whether or not this set is empty.
   *
   * This method returns true if the set is empty, false otherwise.
   *
   * @return Whether or not the set is empty.
   */
  bool empty() const override;

  //@{
  /**
   * @brief Return pointer to detected object at specified index.
   *
   * Returns a reference to the element at specified location pos,
   * with bounds checking.
   *
   * If pos is not within the range of the container, an exception of
   * type std::out_of_range is thrown.
   *
   * @param pos Position of element to return (from zero).
   *
   * @return Shared pointer to specified element.
   *
   * @throws std::range if position is now within the range of objects
   * in container.
   */
  detected_object_sptr at( size_t pos ) override;
  const detected_object_sptr at( size_t pos ) const override;
  //@}

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
  detected_object_set_sptr select( double threshold = detected_object_type::INVALID_SCORE ) const;

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
  detected_object_set_sptr select( const std::string& class_name,
                                   double             threshold = detected_object_type::INVALID_SCORE ) const;

  /**
   * @brief Filter detections via an arbitrary function
   *
   * Filter the detected object set based on some predicate function
   *
   * @param p Predicate function which returns true if an element should
   *          be removed.
   *
   * @return List of detections.
   */
  template< class UnaryPredicate >
  void filter( UnaryPredicate p )
  {
    m_detected_objects.erase(
      std::remove_if( m_detected_objects.begin(),
                      m_detected_objects.end(),
                      p ),
      m_detected_objects.end() );
  }

  /**
   * @brief Scale all detection locations by some scale factor.
   *
   * This method changes the bounding boxes within all stored detections
   * by scaling them by some scale factor.
   *
   * @param scale Scale factor
   */
  void
  VITAL_DEPRECATED
  scale( double scale_factor );

  /**
   * @brief Shift all detection locations by some translation offset.
   *
   * This method shifts the bounding boxes within all stored detections
   * by a supplied column and row shift.
   *
   * Note: Detections in this set can be shared by multiple sets, so
   * shifting the detections in this set will also shift the detection
   * in other sets that share this detection. If this is going to be a
   * problem, clone() this set before shifting.
   *
   * @param col_shift Column  (a.k.a. x, i, width) translation factor
   * @param row_shift Row (a.k.a. y, j, height) translation factor
   */
  void
  VITAL_DEPRECATED
  shift( double col_shift, double row_shift );

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

protected:
  iterator::next_value_func_t get_iter_next_func() override;
  const_iterator::next_value_func_t get_const_iter_next_func() const override;


private:
  // List of detections ordered by confidence value.
  std::vector< detected_object_sptr > m_detected_objects;

  attribute_set_sptr m_attrs;
};

} } // end namespace

#endif
