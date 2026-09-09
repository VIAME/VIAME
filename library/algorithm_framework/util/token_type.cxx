// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_type.h"

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// Constructor
///
token_type
::token_type( std::string const& name )
  : m_typeName( name )
{}

token_type::
~token_type()
{}

// ----------------------------------------------------------------------------
/// Return token type name.
///
std::string const&
token_type
::token_type_name() const
{
  return m_typeName;
}

} // namespace vital

}   // end namespace
