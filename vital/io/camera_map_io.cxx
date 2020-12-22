// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of camera map io functions
 */

#include "camera_map_io.h"
#include "camera_io.h"
#include <vital/exceptions.h>

#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace vital {

/// Load a camera map from krtd files stored in a directory.
camera_map_sptr
read_krtd_files( std::vector< path_t > const& img_files, path_t const& dir )
{
  if ( ! kwiversys::SystemTools::FileExists( dir ) )
  {
    VITAL_THROW( path_not_exists, dir );
  }

  camera_map::map_camera_t cameras;

  for ( size_t fid = 0; fid < img_files.size(); ++fid )
  {
    try
    {
      cameras[fid] = read_krtd_file( img_files[fid], dir );
    }
    catch ( const file_not_found_exception& )
    {
      continue;
    }
  }

  if ( cameras.empty() )
  {
    VITAL_THROW( invalid_data, "No krtd files found" );
  }

  return camera_map_sptr( new simple_camera_map( cameras ) );
}

} } // end namespace vital
