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

#ifndef VITAL_RANGE_VALID_H
#define VITAL_RANGE_VALID_H

#include <vital/range/defs.h>

namespace kwiver {
namespace vital {
namespace range {

// ----------------------------------------------------------------------------
/// Validity-checking range adapter.
/**
 * This range adapter applies a validity filter to the elements of a range.
 * When iterating over the range, only elements for which \c !!item is \c true
 * will be seen.
 *
 * \par Example:
 \code
 namespace r = kwiver::vital::range;

 std::vector<std::shared_ptr<int>> values = get_values();

 // Won't crash, even if some items are null pointers
 for ( auto const& p : values | r::valid )
   std::cout << *p << std::endl;
 \endcode
 */
template < typename Range >
class valid_view : public generic_view
{
protected:
  using range_iterator_t = typename range_ref< Range >::iterator_t;

public:
  using value_ref_t = typename range_ref< Range >::value_ref_t;

  valid_view( valid_view const& ) = default;
  valid_view( valid_view&& ) = default;

  class iterator
  {
  public:
    iterator() = default;
    iterator( iterator const& ) = default;
    iterator& operator=( iterator const& ) = default;

    bool operator!=( iterator const& other ) const
    { return m_iter != other.m_iter; }

    value_ref_t operator*() const { return *m_iter; }

    iterator& operator++();

    operator bool() const { return m_iter != m_end; }

  protected:
    friend class valid_view;
    iterator( range_iterator_t const& iter,
              range_iterator_t const& end )
      : m_iter{ iter }, m_end{ end } {}

    range_iterator_t m_iter, m_end;
  };

  valid_view( Range&& range ) : m_range( std::forward< Range >( range ) ) {}

  iterator begin() const;

  iterator end() const
  { return { m_range.end(), m_range.end() }; }

protected:
  range_ref< Range > m_range;
};

// ----------------------------------------------------------------------------
template < typename Range >
typename valid_view< Range >::iterator
valid_view< Range >
::begin() const
{
  auto iter = iterator{ m_range.begin(), m_range.end() };
  return ( iter && !!( *iter ) ? iter : ++iter );
}

// ----------------------------------------------------------------------------
template < typename Range >
typename valid_view< Range >::iterator&
valid_view< Range >::iterator
::operator++()
{
  while ( m_iter != m_end )
  {
    ++m_iter;
    if ( m_iter != m_end && !!( *m_iter ) ) break;
  }
  return *this;
}

///////////////////////////////////////////////////////////////////////////////

KWIVER_MUTABLE_RANGE_ADAPTER( valid )

} } } // end namespace

#endif
