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

#ifndef VITAL_RANGE_SLIDING_H
#define VITAL_RANGE_SLIDING_H

#include <vital/range/defs.h>
#include <vital/range/integer_sequence.h>

#include <array>

namespace kwiver {
namespace vital {
namespace range {

// ----------------------------------------------------------------------------
/// Sliding window range adapter.
/**
 * This range adapter provides a sliding view on a range. Given a \c Size, the
 * first iteration will return that number of elements, in the same order as
 * the input range. Each subsequent step will shift this window by one. If the
 * input range has fewer than \c Size elements, the output range will be empty.
 *
 * \par Example:
 * \code
 * namespace r = kwiver::vital::range;
 *
 * std::vector<int> values = { 1, 2, 3, 4, 5 };
 *
 * for ( auto x : values | r::sliding< 3 > )
 *   std::cout << x[0] << x[1] << x[2] << std::endl;
 *
 * // Output:
 * //  123
 * //  234
 * //  345
 * \endcode
 *
 * \note Due to C++ limitations, if \c sliding is applied immediately to a
 *       C-style array, the template parameter list must be followed by
 *       <code>()</code>.
 */
template < size_t Size, typename Range >
class sliding_view : public generic_view
{
protected:
  using range_iterator_t = typename range_ref< Range const >::iterator_t;
  using single_value_t = typename range_ref< Range const >::value_t;

public:
  using value_t = std::array< single_value_t, Size >;

  sliding_view( sliding_view const& ) = default;
  sliding_view( sliding_view&& ) = default;

  class iterator
  {
  public:
    iterator() = default;
    iterator( iterator const& ) = default;
    iterator& operator=( iterator const& ) = default;

    bool operator!=( iterator const& other ) const;

    value_t operator*() const;

    iterator& operator++();

  protected:
    friend class sliding_view;
    iterator( range_iterator_t iter, range_iterator_t const& end );

    struct dereference_helper
    {
      template < size_t... Indices >
      static value_t dereference(
        std::array< range_iterator_t, Size > const& iters,
        integer_sequence_t< size_t, Indices... > )
      {
        return {{ *( iters[ Indices ] )... }};
      }
    };

    std::array< range_iterator_t, Size > m_iter;
  };

  sliding_view( Range&& range ) : m_range( std::forward< Range >( range ) ) {}

  iterator begin() const
  { return { m_range.begin(), m_range.end() }; }

  iterator end() const
  { return { m_range.end(), m_range.end() }; }

protected:
  range_ref< Range const > m_range;
};

// ----------------------------------------------------------------------------
template < size_t Size, typename Range >
sliding_view< Size, Range >::iterator
::iterator( range_iterator_t iter, range_iterator_t const& end )
{
  for ( size_t i = 0; i < Size; ++i )
  {
    m_iter[ i ] = iter;
    if ( iter != end ) ++iter;
  }
}

// ----------------------------------------------------------------------------
template < size_t Size, typename Range >
typename sliding_view< Size, Range >::value_t
sliding_view< Size, Range >::iterator
::operator*() const
{
  auto const indices = make_integer_sequence< size_t, Size >();
  return dereference_helper::dereference( m_iter, indices );
}

// ----------------------------------------------------------------------------
template < size_t Size, typename Range >
bool
sliding_view< Size, Range >::iterator
::operator!=( iterator const& other ) const
{
  return m_iter.back() != other.m_iter.back();
}

// ----------------------------------------------------------------------------
template < size_t Size, typename Range >
typename sliding_view< Size, Range >::iterator&
sliding_view< Size, Range >::iterator
::operator++()
{
  for ( size_t i = 0; i < Size - 1; ++i )
  {
    m_iter[ i ] = m_iter[ i + 1 ];
  }
  ++m_iter.back();

  return *this;
}

///////////////////////////////////////////////////////////////////////////////

KWIVER_RANGE_ADAPTER_TEMPLATE( sliding, ( size_t Size ), ( Size ) )

} // namespace range
} // namespace vital
} // namespace kwiver

#endif
