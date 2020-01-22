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

#ifndef VITAL_RANGE_INDIRECT_H
#define VITAL_RANGE_INDIRECT_H

#include <vital/range/defs.h>

namespace kwiver {
namespace vital {
namespace range {

//-----------------------------------------------------------------------------
/// Indirection range adapter.
/**
 * This range adapter applies a level of indirection. This is typically used
 * to suppress the dereferencing of a container iterator in a range-based
 * \c for loop in order to allow iteration over the container's iterators,
 * rather than the values.
 */
template < typename Range >
class indirect_view : public generic_view
{
protected:
  using range_iterator_t = typename range_ref< Range >::iterator_t;

public:
  using value_t = range_iterator_t;

  indirect_view( indirect_view const& ) = default;
  indirect_view( indirect_view&& ) = default;

  class iterator
  {
  public:
    iterator() = default;
    iterator( iterator const& ) = default;
    iterator& operator=( iterator const& ) = default;

    bool operator!=( iterator const& other ) const
    { return m_iter != other.m_iter; }

    value_t operator*() const { return m_iter; }

    iterator& operator++() { ++m_iter; return *this; }

  protected:
    friend class indirect_view;
    iterator( range_iterator_t const& iter ) : m_iter{ iter } {}

    range_iterator_t m_iter;
  };

  indirect_view( Range&& range ) : m_range( std::forward< Range >( range ) ) {}

  iterator begin() const { return { m_range.begin() }; }
  iterator end() const { return { m_range.end() }; }

protected:
  range_ref< Range const > m_range;
};

///////////////////////////////////////////////////////////////////////////////

KWIVER_MUTABLE_RANGE_ADAPTER( indirect )

} } } // end namespace

#endif
