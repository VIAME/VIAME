/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
  ///@}
};

} // end namespace: vital
} // end namespace: kwiver

#endif //KWIVER_VITAL_SET_H_
