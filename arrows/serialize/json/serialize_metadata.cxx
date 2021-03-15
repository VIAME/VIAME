// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of metadata load/save functionality.

#include <arrows/serialize/json/metadata.h>
#include <arrows/serialize/json/serialize_metadata.h>

#include <vital/types/metadata_map.h>

#include <vital/exceptions/io.h>

#include <fstream>
#include <iostream>

namespace kwiver {

namespace arrows {

namespace serialize {

namespace json {

// ----------------------------------------------------------------------------
class serialize_metadata::priv
{
public:
  metadata serializer;
};

// ----------------------------------------------------------------------------
serialize_metadata
::serialize_metadata()
{
}

// ----------------------------------------------------------------------------
serialize_metadata
::~serialize_metadata()
{
}

// ----------------------------------------------------------------------------
kwiver::vital::metadata_map_sptr
serialize_metadata
::load_( std::string const& filename ) const
{
  vital::metadata_map::map_metadata_t metadata_map;

  std::ifstream fin{ filename };

  if( fin )
  {
    std::stringstream buffer;
    buffer << fin.rdbuf();

    auto const& input_string = buffer.str();

    metadata_map = d_->serializer.deserialize_map( input_string );
  }
  else
  {
    VITAL_THROW( vital::file_not_read_exception, filename,
                 "Insufficient permissions or moved file" );
  }

  return std::make_shared< vital::simple_metadata_map >( metadata_map );
}

// ----------------------------------------------------------------------------
void
serialize_metadata
::save_( std::string const& filename,
         vital::metadata_map_sptr data ) const
{
  std::ofstream fout{ filename };

  if( fout )
  {
    auto metadata = data->metadata();
    std::shared_ptr< std::string > serialized =
      d_->serializer.serialize_map( metadata );
    fout << *serialized << std::endl;
  }
  else
  {
    VITAL_THROW( vital::file_write_exception, filename,
                 "Insufficient permissions or moved file" );
  }
}

} // namespace json

} // namespace serialize

} // namespace arrows

} // namespace kwiver
