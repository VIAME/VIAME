// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "location_info.h"

#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace vital {
namespace logger_ns {

typedef kwiversys::SystemTools ST;

/**
   When location information is not available the constant
   <code>NA</code> is returned. Current value of this string
   constant is <b>?</b>.  */
const char* const location_info::NA = "?";
const char* const location_info::NA_METHOD = "?::?";

// ----------------------------------------------------------------
/** Constructor.
 *
 * The default constructor creates a location with all fields set to
 * the "unknown" state.
 */
location_info
::location_info()
  : m_fileName(location_info::NA),
    m_methodName(location_info::NA_METHOD),
    m_lineNumber(-1)
{ }

// ----------------------------------------------------------------
/** Constructor.
 *
 * This constructor creates a location object with a fully described
 * location.
 */
location_info
::location_info (char const* filename, char const* method, int line )
  : m_fileName(filename),
    m_methodName(method),
    m_lineNumber(line)
{ }

// ----------------------------------------------------------------
std::string location_info
::get_file_name() const
{
  return ST::GetFilenameName( m_fileName );
}

// ----------------------------------------------------------------
std::string location_info
::get_file_path() const
{
  return ST::GetFilenamePath( m_fileName );
}

// ----------------------------------------------------------------
std::string location_info
::get_signature() const
{
  return m_methodName;
}

// ----------------------------------------------------------------
std::string location_info
::get_method_name() const
{
  std::string tmp(m_methodName);

  // Clear all parameters from signature
  size_t parenPos = tmp.find('(');
  if (parenPos != std::string::npos)
  {
    tmp.erase(parenPos);
  }

  size_t colonPos = tmp.rfind("::");
  if (colonPos != std::string::npos)
  {
    tmp.erase(0, colonPos + 2);
  }

  size_t spacePos = tmp.rfind(' ');
  if (spacePos != std::string::npos)
  {
    tmp.erase(0, spacePos + 1);
  }

  return ( tmp );
}

// ----------------------------------------------------------------
std::string location_info
::get_class_name() const
{
  std::string tmp(m_methodName);

  // Clear all parameters from signature
  size_t parenPos = tmp.find('(');
  if (parenPos != std::string::npos)
  {
    tmp.erase(parenPos);
  }

  // Erase return type if any
  size_t spacePos = tmp.rfind(' ');
  if (spacePos != std::string::npos)
  {
    tmp.erase(0, spacePos + 1);
  }

  // erase all characters after last "::"
  size_t colonPos = tmp.rfind("::");
  if (colonPos != std::string::npos)
  {
    tmp.erase(colonPos);
  }
  else
  {
    // no class if no "::"
    tmp.clear();
  }

  return ( tmp );
}

// ----------------------------------------------------------------
int location_info
::get_line_number() const
{
  return m_lineNumber;
}

} } } // end namespace
