// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_OVERLOAD_H
#define KWIVER_VITAL_OVERLOAD_H

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
template < class... Args, class T, class R >
constexpr inline
auto
overload( R ( T::* m )( Args... ) ) -> decltype( m )
{ return m; }

// ----------------------------------------------------------------------------
template < class T, class R >
constexpr inline
auto
overload( R ( T::* m )() ) -> decltype( m )
{ return m; }

} // namespace vital

} // namespace kwiver

#endif
