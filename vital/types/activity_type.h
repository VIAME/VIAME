/*ckwg +30
 * Copyright 2020 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

/**
 * \file
 * \brief Header for an activity type
 */

#ifndef VITAL_TYPES_ACTIVITY_TYPE_H_
#define VITAL_TYPES_ACTIVITY_TYPE_H_

#include <vital/vital_export.h>
#include <vital/vital_types.h>

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <utility>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Activity type classification.
 *
 * This class represents a set of possible types for the activity.
 * A type for an activity is represented by a class_name string
 * and a score.
 *
 * When an activity is classified, there may be several possibilities
 * determined, since the classification process is probabilistic. This
 * class captures the set of possible types along with the relative
 * likelihood or score.
 *
 * Note that score values in are *not* constrained to
 * [0.0,1.0] because different detectors use different approaches for
 * scores. These scores can be normalized, but that is up to the user
 * of these values.
 */

class VITAL_EXPORT activity_type
{
public:
  using class_map_t = std::map< activity_label_t const, activity_confidence_t >;
  using class_const_iterator_t = class_map_t::const_iterator;

  /**
   * @brief Create an empty activity type.
   *
   * An object is created without class_names or scores.
   */
  activity_type();

  /**
   * @brief Create an activity type with multiple class names and scores.
   *
   * Create a new activity type instance with a set of labels and
   * likelihoods. The parameters have corresponding ordering, which
   * means that the first label is for the first likelihood , and so
   * on.
   *
   * The number of elements in the parameter vectors must be the same.
   *
   * @param class_names List of names for the possible classes.
   * @param scores Vector of scores for this activity type.*
   * @throws std::invalid_argument if the vector lengths differ
   */
  activity_type( const std::vector< activity_label_t >& class_names,
                 const std::vector< activity_confidence_t >& scores );

  /**
   * @brief Create new activity type a class and a score.
   *
   * Create a new activty type instance from a single class name
   * and label.
   *
   * @param class_name Class name
   * @param score Probability score for the class
   */
  activity_type( const activity_label_t& class_name,
                 activity_confidence_t score );

  /**
   * @brief Determine if class-name is present.
   *
   * This method determines if the specified class name is present in
   * this activity.
   *
   * @param class_name Class name to test.
   *
   * @return \b true if class name is present.
   */
  bool has_class_name( const activity_label_t& class_name ) const;

  /**
   * @brief Get score for specific class_name.
   *
   * This method returns the score for the specified class_name.  If
   * the name is not associated with this activity type, an exception is
   * thrown.
   *
   * @param class_name Return score for this entry.
   *
   * @throws std::runtime_error If supplied class_name is not
   * associated with this object.
   *
   * @return Score for selected class_name.
   */
  double score( const activity_label_t& class_name ) const;

  /**
   * @brief Get class name with the highest score.
   *
   * This method returns the class name with the highest score.
   *
   * If there are no scores associated with activity type, then an
   * exception is thrown
   *
   * @return Class with highest score.
   *
   * @throws std::runtime_error If no scores are associated with this
   *                            activity type.
   */
  const activity_label_t get_most_likely_class( ) const;

  /**
   * @brief Get class name along with the score of the class with highest score
   *
   * This method returns a pair containing the class name and the score of the
   * class with the highest score.
   *
   * If there are no scores associated with this activity type, then an
   * exception is thrown
   *
   * @return A pair with the class name as the first element and the score as
   *         the second element
   *
   * @throws std::runtime_error If no scores are associated with this
   *                            activity type.
   */
  std::pair<activity_label_t, activity_confidence_t>
    get_most_likely_class_and_score( ) const;

  /**
   * @brief Set score for a class.
   *
   * This method sets or updates the score for a type name. Note that
   * the score value is *not* constrained to [0.0,1.0].
   *
   * If the class_name specified is not previously associated with
   * this object type, it is added, If it is present, the score is
   * updated.
   *
   * @param class_name Class name.
   * @param score Score value for class_name
   */
  void set_score( const activity_label_t& class_name, activity_confidence_t score );

  /**
   * @brief Remove score and class_name.
   *
   * This method removes the type entry for the specified
   * class_name. An exception is thrown if this activity type
   * does not have that class_name.
   *
   * @param label Class name to remove.
   *
   * @throws std::runtime_error If supplied class_name is not
   * associated with this object.
   */
  void delete_score( const activity_label_t& class_name );

  /**
   * @brief Get list of class_names for this detection.
   *
   * This method returns a vector of class_names that apply to the
   * activity type with a score that is greater than or equal to the threshold.
   * The names are ordered by decreasing score.
   *
   * @param threshold Labels with a score below this value are omitted from
   *                  the returned list.
   *
   * @return Ordered list of class_names. Note that the list may be empty.
   */
  std::vector< activity_label_t > class_names( activity_confidence_t threshold ) const;

  /**
   * @brief Get number of class names on this activity type.
   *
   * This method returns the number of class names that are in this
   * activity type.
   *
   * @return Number of registered class names.
   */
  size_t size() const;

  /**
   * @brief Get start iterator to all class/score pairs.
   *
   * This method returns an iterator that may be used to iterate over all
   * class/score pairs. The order in which items will be seen by this iterator
   * is unspecified.
   *
   * @return Start iterator to all class/score pairs.
   *
   * @sa end
   */
  class_const_iterator_t begin() const;

  /**
   * @brief Get end iterator to all class/score pairs.
   *
   * This method returns an iterator that may be used to end iteration over all
   * class/score pairs.
   *
   * @return End iterator to all class/score pairs.
   *
   * @sa begin
   */
  class_const_iterator_t end() const;

  /**
   * @brief Get list of all class_names in use.
   *
   * This method returns an ordered vector of all class_name strings.
   * This set of strings represents the superset of all class_names
   * used to classify objects. Strings are added to this set when a
   * previously unseen class_name is passed to the CTOR or
   *
   * @return Vector of class names.
   */
  std::vector < activity_label_t > all_class_names();

private:
  /**
   * Set of possible classes for this object.
   *
   * This map represents the ordered set of possibilities for this
   * object along with the class names.
   */
  class_map_t m_classes;

};

// typedef for a object_type shared pointer
typedef std::shared_ptr< activity_type > activity_type_sptr;

} }

#endif
