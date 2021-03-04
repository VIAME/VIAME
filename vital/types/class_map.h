// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for class_map class
 */

#ifndef VITAL_CLASS_MAP_H_
#define VITAL_CLASS_MAP_H_

#include <vital/signal.h>

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief Map of classifications to confidence scores for an object.
 *
 * This class represents a set of possible types for an object.
 * A type for an object is represented by a class_name string
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
template < typename T >
class class_map
{
public:
  static constexpr double INVALID_SCORE = std::numeric_limits< double >::min();

  using class_map_t = std::map< std::string const*, double >;
  using class_const_iterator_t = class_map_t::const_iterator;

  /**
   * @brief Create an empty object.
   *
   * An object is created without class_names or scores.
   */
  class_map();

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
  class_map( const std::vector< std::string >& class_names,
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
  class_map( const std::string& class_name,
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
   * the name is not associated with this object, an exception is
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
   * If there are no scores associated with this object, then an
   * exception is thrown
   *
   * @param[out] max_name Class name with the maximum score.
   *
   * @throws std::runtime_error If no scores are associated with this
   *                            object.
   */
  void get_most_likely( std::string& max_name ) const;

  /**
   * @brief Get max score and name.
   *
   * This method returns the maximum score or the most likely class
   * for this object. The score value and class_name are returned.
   *
   * If there are no scores associated with this object, then an
   * exception is thrown
   *
   * @param[out] max_name Class name with the maximum score.
   * @param[out] max_score maximum score
   *
   * @throws std::runtime_error If no scores are associated with this
   *                            object.
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
   * class_name. An exception is thrown if this object type
   * does not have that class_name.
   *
   * @param label Class name to remove.
   *
   * @throws std::runtime_error If supplied class_name is not
   * associated with this object.
   */
  void delete_score( const std::string& class_name );

  /**
   * @brief Get list of class_names for this object.
   *
   * This method returns a vector of class_names that apply to this
   * object. The names are ordered by decreasing score. If an
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

  //@{
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
  class_const_iterator_t cbegin() const;
  //@}

  //@{
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
  class_const_iterator_t cend() const;
  //@}

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

  /**
   * @brief Signal emitted when a new type name is created.
   *
   * This signal is emitted whenever a new type name is seen for the first
   * time. Applications which need to perform some function when this occurs
   * may do so by connecting to this signal. The name of the new type is passed
   * to the slot.
   *
   * @warning
   * Connected slots execute on whichever thread caused the creation of a new
   * detected object type name. This thread should generally be treated as
   * arbitrary, and the slot coded accordingly. Note also that it may be
   * important for performance that the slot does not take a long time to
   * execute.
   */
  static signal< std::string const& > class_name_added;

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
  static std::unordered_set< std::string > s_master_name_set;

  // Used to control access to shared resources
  static std::mutex s_table_mutex;
};

} // namespace vital

} // namespace kwiver

#endif
