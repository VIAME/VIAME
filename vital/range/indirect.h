// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
