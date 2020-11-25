// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of load/save functionality.
 */
#include <vital/exceptions/io.h>
#include <vital/types/metadata_map.h>

#include <arrows/serialize/json/metadata.h>
#include <arrows/serialize/json/serialize_metadata.h>

#include <iostream>
#include <fstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

/// Private implementation class
class serialize_metadata::priv
{
public:
  /// Constructor
  priv() {}

  kwiver::arrows::serialize::json::metadata serializer;
};

/// Constructor
serialize_metadata
::serialize_metadata()
{ }

/// Destructor
serialize_metadata
::~serialize_metadata()
{ }

/// Load in the data from a filename.
/**
 * \throws file_not_read_exception
 *    Thrown if the file can't be opened for reading, likely due to
 *    permissions or not being present.
 *
 * \param filename the path to the file the load
 * \returns the metadata map for a video
 */
kwiver::vital::metadata_map_sptr
serialize_metadata::load_(std::string const& filename) const
{
  kwiver::vital::metadata_map::map_metadata_t metadata_map;

  std::ifstream fin(filename);

  if ( fin )
  {
    std::stringstream buffer;
    buffer << fin.rdbuf();
    std::string input_string = buffer.str();

    metadata_map = d_->serializer.deserialize_map(input_string);
  }
  else
  {
    VITAL_THROW(kwiver::vital::file_not_read_exception, filename,
                "Insufficient permissions or moved file");
  }

  auto metadata_map_ptr = std::make_shared< kwiver::vital::simple_metadata_map >(
      kwiver::vital::simple_metadata_map( metadata_map ) );
  return metadata_map_ptr;
}

/// Save metadata to a file.
/**
 * \throws file_write_exception
 *    Thrown if the file can't be opened, likely due to permissions
 *    or a missing containing directory
 *
 * \param filename the path to the file to save
 * \param data the metadata map for a video
 */
void
serialize_metadata::save_(std::string const& filename,
                          kwiver::vital::metadata_map_sptr data) const
{
  std::ofstream fout( filename.c_str() );

  if( fout )
  {
    auto metadata = data->metadata();
    std::shared_ptr< std::string > serialized =
      d_->serializer.serialize_map(metadata);
    fout << *serialized << std::endl;
  }
  else
  {
    VITAL_THROW(kwiver::vital::file_write_exception, filename,
                "Insufficient permissions or moved file");
  }
}

} } } } // end namespace
