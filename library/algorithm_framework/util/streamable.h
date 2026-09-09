// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file \brief This file contains the type trait `is_streamable`,
// which is true if the template parameter can be streamed
// in `std::ostream`.

#ifndef KWIVER_VITAL_UTIL_STREAMABLE_H
#define KWIVER_VITAL_UTIL_STREAMABLE_H

#include <type_traits>

namespace kwiver::vital::streamable {

template < typename T, typename = void >
struct is_streamable : std::false_type {};

template < typename T >
struct is_streamable< T,
  std::void_t< decltype( std::declval< std::ostream& >() << std::declval< const T& >() ) > >: std::true_type {};

} // namespace kwiver::vital::streamable

#endif
