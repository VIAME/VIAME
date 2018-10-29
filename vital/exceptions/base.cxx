/*ckwg +29
 * Copyright 2013-2018 by Kitware, Inc.
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
