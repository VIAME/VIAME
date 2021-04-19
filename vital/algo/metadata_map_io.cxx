// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of load/save wrapping functionality.
 */

#include "metadata_map_io.h"

#include <vital/algo/algorithm.txx>
#include <vital/exceptions/io.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <kwiversys/SystemTools.hxx>

#include <fstream>

/// \cond DoxygenSuppress
INSTANTIATE_ALGORITHM_DEF( kwiver::vital::algo::metadata_map_io );
/// \endcond

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------
metadata_map_io
::metadata_map_io()
{
  attach_logger( "algo.metadata_map_io" );
}

// ----------------------------------------------------------------------------
metadata_map_sptr
metadata_map_io
::load( std::string const& filename ) const
{
  // Make sure that the given file path exists and is a file
  if( !kwiversys::SystemTools::FileExists( filename ) )
  {
    VITAL_THROW( path_not_exists, filename );
  }
  else if( kwiversys::SystemTools::FileIsDirectory( filename ) )
  {
    VITAL_THROW( path_not_a_file, filename );
  }

  std::ifstream fin{ filename };

  return load( fin );
}

// ----------------------------------------------------------------------------
metadata_map_sptr
metadata_map_io
::load( std::istream& fin, std::string const& filename ) const
{
  return load_( fin, filename );
}

// ----------------------------------------------------------------------------
void
metadata_map_io
::save( std::string const& filename, metadata_map_sptr data ) const
{
  // Make sure that the given file path's containing directory exists and is
  // actually a directory
  std::string containing_dir =
    kwiversys::SystemTools::GetFilenamePath(
      kwiversys::SystemTools::CollapseFullPath( filename ) );

  if( !kwiversys::SystemTools::FileExists( containing_dir ) )
  {
    VITAL_THROW( path_not_exists, containing_dir );
  }
  else if( !kwiversys::SystemTools::FileIsDirectory( containing_dir ) )
  {
    VITAL_THROW( path_not_a_directory, containing_dir );
  }

  /// \brief Default plugin algorithm registration interface implementation.

  std::ofstream fout{ filename };

  save( fout, data, filename );
}

// ----------------------------------------------------------------------------
void
metadata_map_io
::save( std::ostream& fout, metadata_map_sptr data,
        std::string const& filename ) const
{
  save_( fout, data, filename );
}

// ----------------------------------------------------------------------------
vital::algorithm_capabilities const&
metadata_map_io
::get_implementation_capabilities() const
{
  return this->m_capabilities;
}

// ----------------------------------------------------------------------------
void
metadata_map_io
::set_capability( algorithm_capabilities::capability_name_t const& name,
                  bool val )
{
  this->m_capabilities.set_capability( name, val );
}

// ----------------------------------------------------------------------------
void
metadata_map_io
::set_configuration( VITAL_UNUSED vital::config_block_sptr config )
{
}

// ----------------------------------------------------------------------------
bool
metadata_map_io
::check_configuration( VITAL_UNUSED vital::config_block_sptr config ) const
{
  return true;
}

} // namespace algo

} // namespace vital

} // namespace kwiver
