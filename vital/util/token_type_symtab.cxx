// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "token_type_symtab.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
token_type_symtab::
token_type_symtab(std::string const& name)
  : token_type( name )
{ }

// ----------------------------------------------------------------
token_type_symtab::
 ~token_type_symtab()
{ }

// ----------------------------------------------------------------
void
token_type_symtab::
add_entry (std::string const& name, std::string const& value)
{
  m_table[name] = value;
}

// ----------------------------------------------------------------
void
token_type_symtab::
remove_entry (std::string const& name)
{
  m_table.erase (name);
}

// ----------------------------------------------------------------
bool
token_type_symtab::
lookup_entry (std::string const& name, std::string& result) const
{
  bool retcode( false );
  result.clear();

  if ( m_table.count( name ) )
  {
    result = m_table.at(name);
    retcode = true;
  }

  return retcode;
}

} } // end namespace
