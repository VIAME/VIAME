// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "location_info.h"

#include <filesystem>

namespace kwiver {

namespace vital {

namespace logger_ns {

/// When location information is not available the constant
/// <code>NA</code> is returned. Current value of this string
/// constant is <b>?</b>.
const char* const location_info::NA = "?";
const char* const location_info::NA_METHOD = "?::?";

// ----------------------------------------------------------------------------
location_info
::location_info()
  : m_fileName( location_info::NA ),
    m_methodName( location_info::NA_METHOD ),
    m_lineNumber( -1 )
{}

// ----------------------------------------------------------------------------
location_info
::location_info( char const* filename, char const* method, int line )
  : m_fileName( filename ),
    m_methodName( method ),
    m_lineNumber( line )
{}

// ----------------------------------------------------------------------------
std::string
location_info
::get_file_name() const
{
  return std::filesystem::path( m_fileName ).filename().string();
}

// ----------------------------------------------------------------------------
std::string
location_info
::get_signature() const
{
  return m_methodName;
}

// ----------------------------------------------------------------------------
int
location_info
::get_line_number() const
{
  return m_lineNumber;
}

} // namespace logger_ns

} // namespace vital

}   // end namespace
