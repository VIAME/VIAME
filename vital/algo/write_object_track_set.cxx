// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of save wrapping functionality.
 */

#include "write_object_track_set.h"

#include <memory>

#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::write_object_track_set);
/// \endcond

namespace kwiver {
namespace vital {
namespace algo {

write_object_track_set
::write_object_track_set()
  : m_stream( 0 )
  , m_stream_owned( false )
{
  attach_logger( "algo.write_object_track_set" );
}

write_object_track_set
::~write_object_track_set()
{
}

// ------------------------------------------------------------------------------------
void
write_object_track_set
::open( std::string const& filename )
{
  // try to open the file
  std::unique_ptr< std::ostream > file( new std::ofstream( filename ) );

  if( ! *file )
  {
    VITAL_THROW( file_not_found_exception, filename, "open failed" );
  }

  m_stream = file.release();
  m_stream_owned = true;
  m_filename = filename;
}

// ------------------------------------------------------------------------------------
void
write_object_track_set
::use_stream( std::ostream* strm )
{
  m_stream = strm;
  m_stream_owned = false;
}

// ------------------------------------------------------------------------------------
void
write_object_track_set
::close()
{
  if( m_stream_owned )
  {
    delete m_stream;
  }

  m_stream = 0;
}

// ------------------------------------------------------------------------------------
std::ostream&
write_object_track_set
::stream()
{
  return *m_stream;
}

// ------------------------------------------------------------------------------------
std::string const&
write_object_track_set
::filename()
{
  return m_filename;
}

} } } // end namespace
