// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_DEMANGLE_H
#define KWIVER_VITAL_DEMANGLE_H

#include <vital/util/vital_util_export.h>

#include <string>
#include <typeinfo>

namespace kwiver {
namespace vital {

VITAL_UTIL_EXPORT std::string demangle( char const* name );
VITAL_UTIL_EXPORT std::string demangle( std::string const& name );

/**
 * @brief Demangle type name from a specific type.
 *
 * Usage:
\code
struct foo { };
foo* foo_ptr = new foo;
std::cout << type_name( foo_ptr ) << std::endl;
\endcode
 */
template <class T>
std::string type_name(const T& t)
{
    return demangle( typeid(t).name() );
}

} } // end namespace

#endif /* KWIVER_VITAL_DEMANGLE_H */
