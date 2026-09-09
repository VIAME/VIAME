// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of load wrapping functionality.

#include "read_object_track_set.h"

#include <memory>

#include <viame/algorithm_framework/exceptions/io.h>
#include <viame/core_types/vital_types.h>

#include <kwiversys/SystemTools.hxx>

namespace kwiver {

namespace vital {

namespace algo {

read_object_track_set
::read_object_track_set()
  : m_stream( nullptr ),
    m_stream_owned( false )
{
  attach_logger( "algo.read_object_track_set" );
}

read_object_track_set
::~read_object_track_set()
{
  if( m_stream && m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = nullptr;
}

// ----------------------------------------------------------------------------
void
read_object_track_set
::open( std::string const& filename )
{
  if( m_stream && m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = nullptr;

  // Make sure that the given file path exists and is a file.
  if( !kwiversys::SystemTools::FileExists( filename ) )
  {
    VITAL_THROW( path_not_exists, filename );
  }

  if( kwiversys::SystemTools::FileIsDirectory( filename ) )
  {
    VITAL_THROW( path_not_a_file, filename );
  }

  // try to open the file
  std::unique_ptr< std::istream > file( new std::ifstream( filename ) );

  if( !*file )
  {
    VITAL_THROW( file_not_found_exception, filename, "open failed" );
  }

  m_stream = file.release();
  m_stream_owned = true;

  new_stream();
}

// ----------------------------------------------------------------------------
void
read_object_track_set
::use_stream( std::istream* strm )
{
  m_stream = strm;
  m_stream_owned = false;

  new_stream();
}

// ----------------------------------------------------------------------------
void
read_object_track_set
::close()
{
  if( m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = nullptr;
}

// ----------------------------------------------------------------------------
bool
read_object_track_set
::at_eof() const
{
  if( m_stream )
  {
    return m_stream->eof();
  }
  else
  {
    return true;
  }
}

// ----------------------------------------------------------------------------
std::istream&
read_object_track_set
::stream()
{
  return *m_stream;
}

// ----------------------------------------------------------------------------
void
read_object_track_set
::new_stream()
{}

} // namespace algo

} // namespace vital

} // namespace kwiver
