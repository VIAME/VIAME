/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation of vital global uid
 */

#include "uid.h"

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
uid::
uid( const std::string& data)
  : m_uid( data )
{
}


uid::
uid( const char* data, size_t byte_count )
  : m_uid( data, byte_count )
{
}


uid::
uid()
{ }


// ------------------------------------------------------------------
bool
uid::
is_valid() const
{
  return ! m_uid.empty();
}


// ------------------------------------------------------------------
std::string const&
uid::
value() const
{
  return m_uid;
}


// ------------------------------------------------------------------
size_t
uid::
size() const
{
  return m_uid.size();
}


// ------------------------------------------------------------------
bool
uid::
operator==( const uid& other ) const
{
  return this->m_uid == other.m_uid;
}


// ------------------------------------------------------------------
bool
uid::
operator!=( const uid& other ) const
{
  return this->m_uid != other.m_uid;
}


// ------------------------------------------------------------------
bool
uid::
operator<( const uid& other ) const
{
  return this->m_uid < other.m_uid ;
}

} } // end namespace
