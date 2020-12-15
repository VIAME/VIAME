// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
