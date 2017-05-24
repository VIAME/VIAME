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

#include "global_uid.h"

#include <uuid/uuid.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
global_uid::
global_uid()

{
  // This may need work to be more system independent.
  uuid_t new_uuid;
  uuid_generate( new_uuid );
  const char* cc = (const char *)&new_uuid[0];

  m_global_uid = std::string( cc, sizeof( new_uuid ));
}

// ------------------------------------------------------------------
global_uid::
global_uid( const std::string& data)
  : m_global_uid( data )
{
}

global_uid::
global_uid( const char* data, size_t byte_count )
  : m_global_uid( data, byte_count )
{
}


// ------------------------------------------------------------------
const char*
global_uid::
value() const
{
  return m_global_uid.data();
}


// ------------------------------------------------------------------
size_t
global_uid::
size() const
{
  return m_global_uid.size();
}


// ------------------------------------------------------------------
bool
global_uid::
operator==( const global_uid& other ) const
{
  return this->m_global_uid == other.m_global_uid;
}


// ------------------------------------------------------------------
bool
global_uid::
operator!=( const global_uid& other ) const
{
  return this->m_global_uid != other.m_global_uid;
}


// ------------------------------------------------------------------
bool
global_uid::
operator<( const global_uid& other ) const
{
  return this->m_global_uid < other.m_global_uid ;
}

} } // end namespace
