/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Implementation of vital uuid
 */

#include "vital_uuid.h"

#include <sstream>
#include <cstring>
#include <iomanip>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
vital_uuid::
vital_uuid( const vital_uuid::uuid_data_t& data)
{
  memcpy( this->m_uuid, data, sizeof( this->m_uuid) );
}


// ------------------------------------------------------------------
  const kwiver::vital::vital_uuid::uuid_data_t&
vital_uuid::
uuid() const
{
  return m_uuid;
}

// ------------------------------------------------------------------
std::string
vital_uuid::
format() const
{
  std::stringstream str;
  size_t idx = 0;
  static constexpr char convert[] = "0123456789abcdef";

#define CONV(B)  str << convert[(B >> 4) & 0x0f] << convert[B & 0x0f]

  for (int i = 0; i < 4; i++, idx++) { CONV(m_uuid[idx]); } str << "-";
  for (int i = 0; i < 2; i++, idx++) { CONV(m_uuid[idx]); } str << "-";
  for (int i = 0; i < 2; i++, idx++) { CONV(m_uuid[idx]); } str << "-";
  for (int i = 0; i < 2; i++, idx++) { CONV(m_uuid[idx]); } str << "-";
  for (int i = 0; i < 6; i++, idx++) { CONV(m_uuid[idx]); }

#undef CONV

  return str.str();
}


bool vital_uuid::
operator==( const vital_uuid& other )
{
  for (int i = 0; i < sizeof(m_uuid); i++)
  {
    if (this->m_uuid[i] != other.m_uuid[i])
    {
      return false;
    }
  }
  return true;
}


bool vital_uuid::
operator!=( const vital_uuid& other )
{
  return ! operator==( other );
}


} } // end namespace
