/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "location_info.h"

#include <boost/filesystem/path.hpp>

namespace kwiver {
namespace logger_ns {

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
  boost::filesystem::path file = m_fileName;
  return file.filename().string();
}


// ----------------------------------------------------------------
std::string location_info
::get_file_path() const
{
  boost::filesystem::path file = m_fileName;
  return file.root_path().string();
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

} // end namespace
} // end namespace
