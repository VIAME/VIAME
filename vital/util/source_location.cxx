/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
