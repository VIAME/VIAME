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

#include "timestamp.h"

#include <sstream>
#include <string>
#include <cstring>
#include <ctime>

namespace kwiver {
namespace vital {

timestamp::timestamp()
  : m_valid_time( false ),
    m_valid_frame( false ),
    m_time( 0 ),
    m_frame( 0 )
{ }


timestamp::timestamp( time_t t, frame_t f )
  : m_valid_time( true ),
    m_valid_frame( true ),
    m_time( t ),
    m_frame( f )
{ }


timestamp& timestamp
::set_time_usec( time_t t )
{
  m_time = t;
  m_valid_time = true;

  return *this;
}


timestamp& timestamp
::set_frame( frame_t f)
{
  m_frame = f;
  m_valid_frame = true;

  return *this;
}


timestamp& timestamp
::set_invalid()
{
  m_valid_time = false;
  m_valid_frame = false;

  return *this;
}


double timestamp
::get_time_seconds() const
{
  return static_cast< double > (m_time) * 1e-6; // convert from usec to sec
}


std::string timestamp
::pretty_print() const
{
  std::stringstream str;
  std::string c_tim( "" );
  std::time_t tt = static_cast< std::time_t > ( this->get_time_seconds() );

  std::streamsize old_prec = str.precision();
  str.precision(16);

  str << "ts(f: ";

  if ( this->has_valid_frame() )
  {
      str << this->get_frame();
  }
  else
  {
    str << "<inv>";
  }

  str << ", t: ";

  if ( this->has_valid_time() )
  {
    char* p = ctime( &tt ); // this may return null if <tt> is out of range,
    if ( p )
    {
      char buffer[128];
      c_tim = " (";
      buffer[0] = 0;
      strncpy( buffer, p, sizeof buffer );
      buffer[std::strlen( buffer ) - 1] = 0; // remove NL

      c_tim = c_tim + buffer;
      c_tim = c_tim + ")";

      str << this->get_time_usec() << c_tim;
    }
    else
    {
      str << " (time " << tt << " out of bounds?)";
    }
  }
  else
  {
    str << "<inv>";
  }

  str << ")";

  str.precision( old_prec );
  return str.str();
}

} } // end namespace
