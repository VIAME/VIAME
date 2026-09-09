// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of load/save wrapping functionality.

#include "detected_object_set_output.h"

#include <memory>

#include <viame/algorithm_framework/exceptions/io.h>
#include <viame/core_types/vital_types.h>

#include <kwiversys/SystemTools.hxx>

namespace kwiver {

namespace vital {

namespace algo {

detected_object_set_output
::detected_object_set_output()
  : m_stream( nullptr ),
    m_stream_owned( false )
{
  attach_logger( "algo.detected_object_set_output" );
}

detected_object_set_output
::~detected_object_set_output()
{
  close();
}

// ----------------------------------------------------------------------------
void
detected_object_set_output
::open( std::string const& filename )
{
  // try to open the file
  std::unique_ptr< std::ostream > file( new std::ofstream( filename ) );
  if( !*file )
  {
    VITAL_THROW( file_not_found_exception, filename, "open failed" );
  }

  m_stream = file.release();
  m_stream_owned = true;
  m_filename = filename;
}

// ----------------------------------------------------------------------------
void
detected_object_set_output
::use_stream( std::ostream* strm )
{
  m_stream = strm;
  m_stream_owned = false;
}

// ----------------------------------------------------------------------------
void
detected_object_set_output
::close()
{
  if( m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = nullptr;
}

// ----------------------------------------------------------------------------
std::ostream&
detected_object_set_output
::stream()
{
  return *m_stream;
}

// ----------------------------------------------------------------------------
std::string const&
detected_object_set_output
::filename()
{
  return m_filename;
}

} // namespace algo

} // namespace vital

} // namespace kwiver
