// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VITAL base exception implementation
 */

#include "base.h"
#include <sstream>

namespace kwiver {
namespace vital {

vital_exception
::vital_exception() noexcept
  : std::exception()
  , m_line_number(0)
{
}

vital_exception
::~vital_exception() noexcept
{
}

// ------------------------------------------------------------------
void
vital_exception
::set_location( std::string const& file, int line )
{
  m_file_name = file;
  m_line_number = line;
}

// ------------------------------------------------------------------
char const*
vital_exception
::what() const noexcept
{
  std::ostringstream sstr;
  sstr << m_what;

  if ( ! m_file_name.empty() )
  {
    sstr << ", thrown from " << m_file_name << ":" << m_line_number;
  }

  m_what_loc = sstr.str();

  return this->m_what_loc.c_str();
}

// ==================================================================
invalid_value
::invalid_value( std::string reason ) noexcept
{
  m_what = "Invalid value(s): " + reason;
}

invalid_value
::~invalid_value() noexcept
{
}

} } // end namespace vital
