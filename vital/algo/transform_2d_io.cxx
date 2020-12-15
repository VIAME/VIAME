// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of load/save wrapping functionality.
 */

#include "transform_2d_io.h"

#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF(kwiver::vital::algo::transform_2d_io);
/// \endcond

namespace kwiver {
namespace vital {
namespace algo {

// ----------------------------------------------------------------------------
transform_2d_io
::transform_2d_io()
{
  attach_logger( "algo.transform_2d_io" );
}

// ----------------------------------------------------------------------------
transform_2d_sptr
transform_2d_io
::load( std::string const& filename ) const
{
  // Make sure that the given file path exists and is a file.
  if ( ! kwiversys::SystemTools::FileExists( filename ) )
  {
    VITAL_THROW( path_not_exists, filename );
  }
  else if ( kwiversys::SystemTools::FileIsDirectory( filename ) )
  {
    VITAL_THROW( path_not_a_file, filename );
  }

  return this->load_( filename );
}

// ----------------------------------------------------------------------------
void
transform_2d_io
::save( std::string const& filename, transform_2d_sptr data ) const
{
  // Make sure that the given file path's containing directory exists and is
  // actually a directory.
  std::string containing_dir = kwiversys::SystemTools::GetFilenamePath(
    kwiversys::SystemTools::CollapseFullPath( filename ) );

  if ( ! kwiversys::SystemTools::FileExists( containing_dir ) )
  {
    VITAL_THROW( path_not_exists, containing_dir );
  }
  else if ( ! kwiversys::SystemTools::FileIsDirectory( containing_dir ) )
  {
    VITAL_THROW( path_not_a_directory, containing_dir );
  }

  this->save_( filename, data );
}

} // namespace algo
} // namespace vital
} // namespace kwiver
