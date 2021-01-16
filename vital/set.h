// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * @file
 * @brief Vital generic set interface.
 */

#ifndef KWIVER_VITAL_SET_H_
#define KWIVER_VITAL_SET_H_

#include <vital/iterator.h>

namespace kwiver {
namespace vital {

/**
 * @brief Mixin set interface for VITAL.
 *
 * Vital sets are intended to be loosely similar to std::set in concept.
 * Vital sets are ordered containers of a single type that:
 *   - are iterable (\see vital::iterable)
 *   - indexable (see the set::at methods)
 *   - can report its size.
 *
 * @tparam T type of elements contained.
 */
template < typename T >
class set
  : public iterable< T >
{
public:
  using value_type = T;

  virtual ~set() = default;

  /**
   * Get the number of elements in this set.
   *
   * @returns Number of elements in this set.
   */
  virtual size_t size() const = 0;

  /**
   * Whether or not this set is empty.
   *
   * @return True if this set is empty or false otherwise.
   */
  virtual bool empty() const = 0;

  //@{
  /**
   * Get the element at specified index.
   *
   * Returns a reference to the element at specified location index,
   * with bounds checking.
   *
   * If index is not within the range of the container, an exception of
   * type std::out_of_range is thrown.
   *
   * @param index Position of element to return (from zero).
   *
   * @return Shared pointer to specified element.
   *
   * @throws std::out_of_range if position is now within the range of objects
   * in container.
   */
  virtual T at( size_t index ) = 0;
  virtual T const at( size_t index ) const = 0;

  T operator[]( size_t index ) { return this->at( index ); }
  const T operator[]( size_t index ) const { return this->at( index ); }
  ///@}
};

} // end namespace: vital
} // end namespace: kwiver

#endif //KWIVER_VITAL_SET_H_
