/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
