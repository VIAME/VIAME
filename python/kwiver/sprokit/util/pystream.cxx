// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "pystream.h"

#include <pybind11/pybind11.h>

#include <algorithm>
#include <string>

#include <cstddef>

namespace sprokit {

namespace python {

// ----------------------------------------------------------------------------
// pyistreambuf implementation

pyistreambuf
::pyistreambuf( pybind11::object const& obj, std::size_t buffer_size )
  : m_obj( obj ),
    m_buffer( buffer_size )
{
  // Set buffer pointers to indicate empty buffer
  char* end = m_buffer.data() + m_buffer.size();
  setg( end, end, end );
}

pyistreambuf
::~pyistreambuf()
{}

pyistreambuf::int_type
pyistreambuf
::underflow()
{
  if( gptr() < egptr() )
  {
    return traits_type::to_int_type( *gptr() );
  }

  pybind11::gil_scoped_acquire acquire;
  ( void ) acquire;

  pybind11::str const bytes =
    pybind11::str( m_obj.attr("read")( m_buffer.size() ) );
      pybind11::ssize_t const sz = len( bytes );

      if( sz == 0 )
      {
        return traits_type::eof();
      }

      std::string const cppstr = bytes.cast< std::string >();
      std::copy( cppstr.begin(), cppstr.end(), m_buffer.data() );

  setg( m_buffer.data(), m_buffer.data(), m_buffer.data() + sz );

  return traits_type::to_int_type( *gptr() );
}

// ----------------------------------------------------------------------------
// pyistream implementation

pyistream
::pyistream( pybind11::object const& obj )
  : std::istream( nullptr ),
    m_buf( obj )
{
  rdbuf( &m_buf );
}

pyistream
::~pyistream()
{}

// ----------------------------------------------------------------------------
// pyostreambuf implementation

pyostreambuf
::pyostreambuf( pybind11::object const& obj, std::size_t buffer_size )
  : m_obj( obj ),
    m_buffer( buffer_size )
{
  // Reserve one character for overflow
  setp( m_buffer.data(), m_buffer.data() + m_buffer.size() - 1 );
}

pyostreambuf
::~pyostreambuf()
{
  sync();
}

pyostreambuf::int_type
pyostreambuf
::overflow( int_type ch )
{
  if( ch != traits_type::eof() )
  {
    *pptr() = static_cast< char >( ch );
    pbump( 1 );
  }

  if( flush_buffer() )
  {
    return ch;
  }

  return traits_type::eof();
}

int
pyostreambuf
::sync()
{
  return flush_buffer() ? 0 : -1;
}

bool
pyostreambuf
::flush_buffer()
{
  std::ptrdiff_t n = pptr() - pbase();

  if( n > 0 )
  {
    pybind11::gil_scoped_acquire acquire;
    ( void ) acquire;

    pybind11::str const bytes( pbase(), static_cast< std::size_t >( n ) );
    m_obj.attr("write")( bytes );
  }

  setp( m_buffer.data(), m_buffer.data() + m_buffer.size() - 1 );

  return true;
}

// ----------------------------------------------------------------------------
// pyostream implementation

pyostream
::pyostream( pybind11::object const& obj )
  : std::ostream( nullptr ),
    m_buf( obj )
{
  rdbuf( &m_buf );
}

pyostream
::~pyostream()
{}

} // namespace python

} // namespace sprokit
