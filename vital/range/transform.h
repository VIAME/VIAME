/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef VITAL_RANGE_TRANSFORM_H
#define VITAL_RANGE_TRANSFORM_H

#include <vital/range/defs.h>

namespace kwiver {
namespace vital {
namespace range {

// ----------------------------------------------------------------------------
/// Transforming range adapter.
/**
 * This range adapter applies a transformation to the elements of a range.
 *
 * \par Example:
 * \code
 * namespace r = kwiver::vital::range;
 *
 * std::vector<int> values = { 1, 2, 3, 4, 5 };
 * auto times_3 = []( int x ){ return x * 3; };
 *
 * for ( auto x : values | r::transform( times_3 ) )
 *   std::cout << x << std::endl;
 *
 * // Output:
 * //  3
 * //  6
 * //  9
 * //  12
 * //  15
 * \endcode
 */
template < typename Functor, typename Range >
class transform_view : public generic_view
{
protected:
  using range_iterator_t = typename range_ref< Range const >::iterator_t;
  using range_value_ref_t = typename range_ref< Range const >::value_ref_t;

public:
  using value_t = typename function_detail< Functor >::return_type_t;
  using transform_function_t = Functor;

  transform_view( transform_view const& ) = default;
  transform_view( transform_view&& ) = default;

  class iterator
  {
  public:
    iterator() = default;
    iterator( iterator const& ) = default;
    iterator& operator=( iterator const& ) = default;

    bool operator!=( iterator const& other ) const
    { return m_iter != other.m_iter; }

    value_t operator*() const { return m_func( *m_iter ); }

    iterator& operator++() { ++m_iter; return *this; }

  protected:
    friend class transform_view;
    iterator( range_iterator_t const& iter,
              transform_function_t const& func )
      : m_iter{ iter }, m_func( func ) {}

    range_iterator_t m_iter;
    transform_function_t m_func;
  };

  transform_view( Range&& range, transform_function_t func )
    : m_range( std::forward< Range >( range ) ), m_func( func ) {}

  iterator begin() const { return { m_range.begin(), m_func }; }
  iterator end() const { return { m_range.end(), m_func }; }

protected:
  range_ref< Range const > m_range;
  transform_function_t m_func;
};

///////////////////////////////////////////////////////////////////////////////

KWIVER_RANGE_ADAPTER_FUNCTION( transform )

} // namespace range
} // namespace vital
} // namespace kwiver

#endif
