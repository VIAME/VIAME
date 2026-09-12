// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of load/save wrapping functionality.

#include "feature_descriptor_io.h"

#include <viame/algorithm_framework/exceptions/io.h>
#include <viame/core_types/vital_types.h>

#include <viame/algorithm_framework/util/file_system.h>

namespace kwiver {

namespace vital {

namespace algo {

feature_descriptor_io
::feature_descriptor_io()
{
  attach_logger( interface_name() );
}

void
feature_descriptor_io
::load(
  std::string const& filename,
  feature_set_sptr& feat,
  descriptor_set_sptr& desc ) const
{
  // Make sure that the given file path exists and is a file.
  if( !kwiver::vital::file_exists( filename ) )
  {
    VITAL_THROW( path_not_exists, filename );
  }
  else if( kwiver::vital::file_is_directory( filename ) )
  {
    VITAL_THROW( path_not_a_file, filename );
  }

  return this->load_( filename, feat, desc );
}

void
feature_descriptor_io
::save(
  std::string const& filename,
  feature_set_sptr feat,
  descriptor_set_sptr desc ) const
{
  // Make sure that the given file path's containing directory exists and is
  // actually a directory.
  std::string containing_dir = kwiver::vital::filename_path(
    kwiver::vital::collapse_full_path( filename ) );

  if( !kwiver::vital::file_exists( containing_dir ) )
  {
    VITAL_THROW( path_not_exists, containing_dir );
  }
  else if( !kwiver::vital::file_is_directory( containing_dir ) )
  {
    VITAL_THROW( path_not_a_directory, containing_dir );
  }
  if( !feat && !desc )
  {
    VITAL_THROW(
      invalid_value,
      "both features and descriptor are Null" );
  }

  this->save_( filename, feat, desc );
}

} // namespace algo

} // namespace vital

} // namespace kwiver
