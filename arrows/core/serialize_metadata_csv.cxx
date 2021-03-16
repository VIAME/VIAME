// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of metadata writing to csv
 */

#include "serialize_metadata_csv.h"

#include <vital/any.h>
#include <vital/exceptions/io.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/geo_point.h>

#include <fstream>
#include <typeinfo>

namespace kv = kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

/// Private implementation class
class serialize_metadata_csv::priv
{
public:
  /// Constructor
  priv() {}

  void write_csv_item( std::string const& csv_field,
                       std::ofstream& fout );
  void write_csv_item( kv::metadata_item const& metadata,
                       std::ofstream& fout );
  void write_csv_header( kv::vital_metadata_tag const& csv_field,
                         std::ofstream& fout );

  kv::metadata_traits md_traits;
};

/// Quote csv field if needed
/**
 * \param csv_field the string representing one data field
 */
//
void
serialize_metadata_csv::priv::
write_csv_item( std::string const& csv_field,
                std::ofstream& fout )
{
  // TODO handle other pathalogical characters such as quotes or newlines
  if (csv_field.find(',') != std::string::npos)
  {
    fout << '"' << csv_field << "\",";
  }
  fout << csv_field << ",";
}

/// Correctly write a metadata item as one or more fields
/**
 * \param metadata one metadata item to be written
 */
//
void
serialize_metadata_csv::priv::
write_csv_item( kv::metadata_item const& metadata,
                std::ofstream& fout )
{
  if( metadata.tag() == kv::VITAL_META_VIDEO_URI )
  {
    // Don't write this internal tag
    return;
  }
  else if( metadata.type() == typeid( kv::geo_point ) )
  {
    // TODO check the CRS and make sure it matches the expected one
    auto const& data = metadata.data();
    kv::geo_point point = kv::any_cast< kv::geo_point >( data );
    kv::geo_point::geo_3d_point_t loc = point.location();

    fout << loc(0) << "," << loc(1) << "," << loc(2) << ",";
  }
  else if( metadata.type() == typeid( kv::geo_polygon ) )
  {
    auto const& data = metadata.data();
    kv::geo_polygon poly = kv::any_cast< kv::geo_polygon >( data );
    kv::polygon verts = poly.polygon();

    for ( size_t n = 0; n < verts.num_vertices(); ++n )
    {
      auto const& v = verts.at( n );
      fout << v[0] << "," << v[1] << ",";
    }
  }
  else if( metadata.type() == typeid( bool ) )
  {
    auto const& data = metadata.data();
    auto const truth = kv::any_cast< bool >( data );
    fout << (truth ? "true," : "false,");
  }
  else if( metadata.type() == typeid( std::string ) )
  {
    auto const& data = metadata.data();
    auto const string = kv::any_cast< std::string >( data );
    fout << '"' << string << "\",";
  }
  else
  {
    write_csv_item( kv::metadata::format_string( metadata.as_string() ),
                    fout );
  }
}

/// Quote csv header item as needed, and explode types as needed
/**
 * \param csv_field the string representing one header item
 */
//
void
serialize_metadata_csv::priv::
write_csv_header( kv::vital_metadata_tag const& csv_field,
                  std::ofstream& fout)
{
  if( csv_field == kv::VITAL_META_SENSOR_LOCATION )
  {
    fout << "\"Sensor Geodetic lon (EPSG:4326)\","
            "\"Sensor Geodetic lat (EPSG:4326)\","
            "\"Sensor Geodetic altitude (meters)\",";
  }
  else if( csv_field == kv::VITAL_META_FRAME_CENTER )
  {
    fout << "\"Geodetic Frame Center lon (EPSG:4326)\","
            "\"Geodetic Frame Center lat (EPSG:4326)\","
            "\"Geodetic Frame Center elevation (meters)\",";
  }
  else if( csv_field == kv::VITAL_META_TARGET_LOCATION )
  {
    fout << "\"Target Geodetic Location lon (EPSG:4326)\","
            "\"Target Geodetic Location lat (EPSG:4326)\","
            "\"Target Geodetic Location elevation (meters)\",";
  }
  else if( csv_field == kv::VITAL_META_CORNER_POINTS )
  {
    fout << "\"Upper left corner point lon (EPSG:4326)\","
            "\"Upper left corner point lat (EPSG:4326)\","
            "\"Upper right corner point lon (EPSG:4326)\","
            "\"Upper right corner point lat (EPSG:4326)\","
            "\"Lower right corner point lon (EPSG:4326)\","
            "\"Lower right corner point lat (EPSG:4326)\","
            "\"Lower left corner point lon (EPSG:4326)\","
            "\"Lower left corner point lat (EPSG:4326)\",";
  }
  else if( csv_field == kv::VITAL_META_VIDEO_URI )
  {
    // Exclude this field as it's long and redundant across frames
    return;
  }
  else
  {
  // Quote all other data
    fout << '"' << md_traits.tag_to_name( csv_field ) << "\",";
  }

}

// ----------------------------------------------------------------------------
serialize_metadata_csv::
serialize_metadata_csv()
  : d_(new priv)
{
  attach_logger( "arrows.core.serialize_metadata_csv" );
}

// ----------------------------------------------------------------------------
serialize_metadata_csv::
~serialize_metadata_csv()
{
}

// ----------------------------------------------------------------------------
kv::metadata_map_sptr
serialize_metadata_csv::load_(std::string const& filename) const
{
  throw kv::file_write_exception(filename, "not implemented");

  kv::metadata_map::map_metadata_t metadata_map;

  auto metadata_map_ptr = std::make_shared< kv::simple_metadata_map >(
      kv::simple_metadata_map( metadata_map ) );
  return metadata_map_ptr;
}

// ----------------------------------------------------------------------------
void
serialize_metadata_csv::save_(std::string const& filename,
                              kv::metadata_map_sptr data) const
{
  std::ofstream fout( filename.c_str() );

  if( !fout )
  {
    VITAL_THROW(kv::file_write_exception, filename,
                "Insufficient permissions or moved file");
  }

  // Accumulate the unique metadata IDs
  std::set<kv::vital_metadata_tag> metadata_ids;
  for ( auto const& frame_data : data->metadata() )
  {
    for ( auto const& metadata_packet : frame_data.second )
    {
      for ( auto const& metadata_item : *metadata_packet )
      {
        kv::vital_metadata_tag type_id = metadata_item.first;
        metadata_ids.insert( type_id );
      }
    }
  }

  // Write out the csv header
  fout << "\"frame ID\",";
  for ( auto const& metadata_id : metadata_ids )
  {
    d_->write_csv_header( metadata_id, fout );
  }
  fout << std::endl;

  for ( auto const& frame_data : data->metadata() )
  {
    for ( auto const& metadata_packet : frame_data.second )
    {
      // Write the frame number
      fout << frame_data.first << ",";
      for ( auto const& metadata_id : metadata_ids )
      {
        if ( metadata_packet->has( metadata_id ) )
        {
          d_->write_csv_item( metadata_packet->find( metadata_id ), fout );
        }
        // exclude this field from writing since it's internal and
        // shared between all frames
        else if ( metadata_id != kv::VITAL_META_VIDEO_URI )
        {
          fout << ",";
        }
      }
      fout << "\n";
    }
  }
  fout << std::flush;
}

} } } // end namespace
