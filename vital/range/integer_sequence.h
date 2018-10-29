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

#ifndef VITAL_RANGE_INTEGER_SEQUENCE_H
#define VITAL_RANGE_INTEGER_SEQUENCE_H

#include <type_traits>

namespace kwiver {
namespace vital {
namespace range {

// ----------------------------------------------------------------------------
template < typename T, T... Values >
struct integer_sequence_t
{
  using type = integer_sequence_t;

  constexpr static auto size = sizeof...( Values );
};

///////////////////////////////////////////////////////////////////////////////
//BEGIN Implementation details

/// \cond DoxygenSuppress

namespace integer_sequence_detail {

// Adapted from https://stackoverflow.com/questions/17424477 and
// https://stackoverflow.com/questions/22486386

// Forward declarations
template < typename S1, typename S2 >
struct concatenator;

template < typename T, T Count, typename = void >
struct generator;

// Type resolvers
template < typename T > using sequence_type = typename T::type;

template < typename S1, typename S2 >
using concat_t = sequence_type< concatenator< S1, S2 > >;

template < typename T, T Count >
using gen_t = sequence_type< generator< T, Count > >;

// Implementations
template < typename T, T... I1, T... I2 >
struct concatenator< integer_sequence_t< T, I1... >,
                     integer_sequence_t< T, I2... > >
  : integer_sequence_t< T, I1..., ( sizeof...( I1 ) + I2 )... >
{};

template < typename T, T Count, typename >
struct generator : concat_t< gen_t< T, Count / 2 >,
                             gen_t< T, Count - ( Count / 2 ) > >
{};

// Boundary cases
template < typename T, T Count >
struct generator< T, Count, typename std::enable_if< Count == 0 >::type >
  : integer_sequence_t< T >
{};

template < typename T, T Count >
struct generator< T, Count, typename std::enable_if< Count == 1 >::type >
  : integer_sequence_t< T, static_cast< T >( 0 ) >
{};

} // end implementation detail namespace

/// \endcond

//END Implementation details
///////////////////////////////////////////////////////////////////////////////

// ----------------------------------------------------------------------------
template < typename T, T Count >
typename integer_sequence_detail::generator< T, Count >::type
make_integer_sequence()
{
  return {};
}

} // namespace range
} // namespace vital
} // namespace kwiver

#endif
