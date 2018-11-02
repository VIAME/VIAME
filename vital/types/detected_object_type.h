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
 * \brief Interface for detected_object_type class
 */

#ifndef VITAL_DETECTED_OBJECT_TYPE_H_
#define VITAL_DETECTED_OBJECT_TYPE_H_

#include <vital/vital_export.h>

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <mutex>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Detected object type classification.
 *
 * This class represents a set of possible types for the detected
 * object. A type for an object is represented by a class_name string
 * and a score.
 *
 * When an object is classified, there may be several possibilities
 * determined, since the classification process is probabilistic. This
 * class captures the set of possible types along with the relative
 * likelyhood or score.
 *
 * Note that score values in this object are *not* constrained to
 * [0.0,1.0] because different detectors use different approaches for
 * scores. These scores can be normalized, but that is up to the user
 * of these values.
 *
 * Note that the list of possible names is managed through a class
 * static string pool. Every effort has been made to make this pool
 * externally unmutable. Your cooperation is appreciated.
 */
class VITAL_EXPORT detected_object_type
{
public:
  static const double INVALID_SCORE;

  using class_map_t = std::map< std::string const*, double >;
  using class_const_iterator_t = class_map_t::const_iterator;

  /**
   * @brief Create an empty object.
   *
   * An object is created without class_names or scores.
   */
  detected_object_type();

  /**
   * @brief Create new object type class.
   *
   * Create a new object type instance with a set of labels and
   * likelyhoods. The parameters have corresponding ordering, which
   * means that the first label is for the first likelyhood , and so
   * on.
   *
   * The number of elements in the parameter vectors must be the same.
   *
   * @param class_names List of names for the possible classes.
   * @param scores Vector of scores for this object.*
   * @throws std::invalid_argument if the vector lengths differ
   */
  detected_object_type( const std::vector< std::string >& class_names,
                        const std::vector< double >& scores );

  /**
   * @brief Create new object type class.
   *
   * Create a new object type instance from a single class name
   * and label.
   *
   * @param class_name Class name
   * @param score Probability score for the class
   */
  detected_object_type( const std::string& class_name,
                        double score );

  /**
   * @brief Determine if class-name is present.
   *
   * This method determines if the specified class name is present in
   * this object.
   *
   * @param class_name Class name to test.
   *
   * @return \b true if class name is present.
   */
  bool has_class_name( const std::string& class_name ) const;

  /**
   * @brief Get score for specific class_name.
   *
   * This method returns the score for the specified class_name.  If
   * the name is associated with this object, an exception is
   * thrown.
   *
   * @param class_name Return score for this entry.
   *
   * @throws std::runtime_error If supplied class_name is not
   * associated with this object.
   *
   * @return Score for selected class_name.
   */
  double score( const std::string& class_name ) const;

  /**
   * @brief Get max class name.
   *
   * This method returns the most likely class for this object.
   *
   * If there are no scores associated with this detection, then an
   * exception is thrown
   *
   * @param[out] max_name Class name with the maximum score.
   *
   * @throws std::runtime_error If no scores are associated with this
   *                            detection.
   */
  void get_most_likely( std::string& max_name ) const;

  /**
   * @brief Get max score and name.
   *
   * This method returns the maximum score or the most likely class
   * for this object. The score value and class_name are returned.
   *
   * If there are no scores associated with this detection, then an
   * exception is thrown
   *
   * @param[out] max_name Class name with the maximum score.
   * @param[out] max_score maximum score
   *
   * @throws std::runtime_error If no scores are associated with this
   *                            detection.
   */
  void get_most_likely( std::string& max_name, double& max_score ) const;

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
  void set_score( const std::string& class_name, double score );

  /**
   * @brief Remove score and class_name.
   *
   * This method removes the type entry for the specified
   * class_name. An exception is thrown if this detected object type
   * does not have that class_name.
   *
   * @param label Class name to remove.
   *
   * @throws std::runtime_error If supplied class_name is not
   * associated with this object.
   */
  void delete_score( const std::string& class_name );

  /**
   * @brief Get list of class_names for this detection.
   *
   * This method returns a vector of class_names that apply to this
   * detection. The names are ordered by decreasing score. If an
   * optional threshold value is supplied, then names with a score
   * not less than that value are included in the returned list.
   *
   * @param threshold If a value is supplied, labels with a score
   *                  below this value are omitted from the returned list.
   *
   * @return Ordered list of class_names. Note that the list may be empty.
   */
  std::vector< std::string > class_names( double threshold = INVALID_SCORE ) const;

  /**
   * @brief Get number of class names on this object.
   *
   * This method returns the number of class names that are in this
   * object type.
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
  static std::vector < std::string > all_class_names();

private:
  const std::string* find_string( const std::string& str ) const;

  /**
   * Set of possible classes for this object.
   *
   * This map represents the ordered set of possibilities for this
   * object along with the class names.
   */
  class_map_t m_classes;

  /**
   * @brief Set of all class_names used.
   *
   * This set of strings represents the superset of all class_names
   * used to classify objects. Strings are added to this set when a
   * previously unseen class_name is passed to the CTOR or
   * set_score().
   */
  static std::set< std::string > s_master_name_set;

  // Used to control access to shared resources
  static std::mutex s_table_mutex;
};

// typedef for a object_type shared pointer
typedef std::shared_ptr< detected_object_type > detected_object_type_sptr;

} }

#endif
