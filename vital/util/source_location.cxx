// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of source_location class.
 */

#include "source_location.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
source_location::
source_location()
  : m_line_num(0)
{ }

// ------------------------------------------------------------------
source_location::
source_location( std::shared_ptr< std::string > f, int l)
  : m_file_name(f)
  , m_line_num(l)
{ }

// ------------------------------------------------------------------
source_location::
source_location( const source_location& other )
  : m_file_name(other.m_file_name)
  , m_line_num( other.m_line_num )
{ }

// ------------------------------------------------------------------
source_location::
~source_location()
{ }

// ------------------------------------------------------------------
std::ostream &
source_location::
format (std::ostream & str) const
{
  if (m_line_num > 0)
  {
    str << *m_file_name << ":" << m_line_num;
  }

  return str;
}

// ------------------------------------------------------------------
bool
source_location::
valid() const
{
  return (  m_line_num > 0) &&
    ( m_file_name ) &&
    ( ! m_file_name->empty() );
}

} } // end namespace
