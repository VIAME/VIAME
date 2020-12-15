// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of load/save wrapping functionality.
 */

#include "detected_object_set_input.h"

#include <memory>

#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::detected_object_set_input);
/// \endcond

namespace kwiver {
namespace vital {
namespace algo {

detected_object_set_input
::detected_object_set_input()
  : m_stream( 0 )
  , m_stream_owned( false )
{
  attach_logger( "algo.detected_object_set_input" );
}

detected_object_set_input
::~detected_object_set_input()
{
  if ( m_stream && m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = 0;
}

// ------------------------------------------------------------------
void
detected_object_set_input
::open( std::string const& filename )
{
    // Make sure that the given file path exists and is a file.
  if ( ! kwiversys::SystemTools::FileExists( filename ) )
  {
    VITAL_THROW( path_not_exists, filename);
  }

  if ( kwiversys::SystemTools::FileIsDirectory( filename ) )
  {
    VITAL_THROW( path_not_a_file, filename);
  }

  // try to open the file
  std::unique_ptr< std::istream > file( new std::ifstream( filename ) );
  if ( ! *file )
  {
    VITAL_THROW( file_not_found_exception, filename, "open failed" );
  }

  m_stream = file.release();
  m_stream_owned = true;

  new_stream();
}

// ------------------------------------------------------------------
void
detected_object_set_input
::use_stream( std::istream* strm )
{
  m_stream = strm;
  m_stream_owned = false;

  new_stream();
}

// ------------------------------------------------------------------
void
detected_object_set_input
::close()
{
  if ( m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = 0;
}

// ------------------------------------------------------------------
bool
detected_object_set_input
::at_eof() const
{
  if ( m_stream )
  {
    return m_stream->eof();
  }
  else
  {
    return true; // really error
  }
}

// ------------------------------------------------------------------
std::istream&
detected_object_set_input
::stream()
{
  return *m_stream;
}

// ------------------------------------------------------------------
void
detected_object_set_input
::new_stream()
{ }

} } } // end namespace
