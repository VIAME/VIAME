/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "timestamp.h"

#include <ostream>
#include <string>
#include <cstring>
#include <ctime>

namespace kwiver
{

timestamp::timestamp()
  : m_valid_time( false ),
    m_valid_frame( false ),
    m_time( 0 ),
    m_frame( 0 )

{
}


timestamp::timestamp( time_t t, frame_t f )
  : m_valid_time( true ),
    m_valid_frame( true ),
    m_time( t ),
    m_frame( f )
{
}


timestamp& timestamp
::set_time( time_t t )
{
  m_time = t;
  m_valid_time = true;
}


timestamp& timestamp
::set_frame( frame_t f)
{
  m_frame = f;
  m_valid_frame = true;
}


timestamp& timestamp
::set_invalid()
{
  m_valid_time = false;
  m_valid_frame = false;
}


std::ostream &
operator<<( std::ostream& str, timestamp const& obj )
{
  std::string c_tim( "" );
  char buffer[128];
  std::time_t tt = static_cast< std::time_t > ( obj.get_time() );

  std::streamsize old_prec = str.precision();
  str.precision(6);

  str << "ts(f: ";

  if ( obj.has_valid_frame() )
  {
      str << obj.get_frame();
  }
  else
  {
    str << "<inv>";
  }

  str << ", t: ";

  if ( obj.has_valid_time() )
  {
    char* p = ctime( &tt ); // this may return null if tt is out of range,
    if ( p )
    {
      c_tim = " (";
      buffer[0] = 0;
      strncpy( buffer, p, sizeof buffer );
      buffer[std::strlen( buffer ) - 1] = 0; // remove NL

      c_tim = c_tim + buffer;
      c_tim = c_tim + ")";

      str << obj.get_time() << c_tim;
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
  return str;
}

} // end namespace

