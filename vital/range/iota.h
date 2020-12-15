// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_RANGE_IOTA_H
#define VITAL_RANGE_IOTA_H

namespace kwiver {

namespace vital {

namespace range {

/**
 * \file
 * \brief Utility to produce a half-open range of integers.
 */

// ----------------------------------------------------------------------------
template < typename T > class iota_range
{
public:
  class iterator;

  iota_range( T count ) : end_{ count } {}

  iterator begin() const { return { T{ 0 } }; }
  iterator end() const { return { end_ }; }

protected:
  T end_;
};

// ----------------------------------------------------------------------------
template < typename T > class iota_range< T >::iterator
{
public:
  T operator*() const { return value_; }
  iterator& operator++() { ++value_; return *this; }

  bool operator==( iterator const& other ) const
  { return value_ == other.value_; }

  bool operator!=( iterator const& other ) const
  { return value_ != other.value_; }

protected:
  friend class iota_range< T >;
  iterator( T value ) : value_{ value } {}

  T value_;
};

// ----------------------------------------------------------------------------
template < typename T >
iota_range< T >
iota( T upper_bound )
{
  return { upper_bound };
}

} // namespace range

} // namespace vital

} // namespace kwiver

#endif
