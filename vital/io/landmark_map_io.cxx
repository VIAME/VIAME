// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of file IO functions for a \ref kwiver::vital::landmark_map
 *
 * Uses the PLY file format
 */

#include "landmark_map_io.h"

#include <vital/exceptions.h>
#include <kwiversys/SystemTools.hxx>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>

namespace kwiver {
namespace vital {

/// Output the given \c landmark_map object to the specified PLY file path
void
write_ply_file( landmark_map_sptr const&  landmarks,
                path_t const&             file_path )
{
  // If the landmark map is empty, throw
  if ( ! landmarks || ( landmarks->size() == 0 ) )
  {
    VITAL_THROW( file_write_exception, file_path,
         "No landmarks in the given landmark map!" );
  }

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_write_exception, file_path,
         "Path given is a directory, can not write file." );
  }

  // Check that the directory of the given filepath exists, creating necessary
  // directories where needed.
  std::string parent_dir =  kwiversys::SystemTools::GetFilenamePath(
    kwiversys::SystemTools::CollapseFullPath( file_path ) );
  if ( ! kwiversys::SystemTools::FileIsDirectory( parent_dir ) )
  {
    if ( ! kwiversys::SystemTools::MakeDirectory( parent_dir ) )
    {
      VITAL_THROW( file_write_exception, parent_dir,
            "Attempted directory creation, but no directory created! No idea what happened here..." );
    }
  }

  // open output file and write the tracks
  std::ofstream ofile( file_path.c_str() );
  // write the PLY header
  ofile << "ply\n"
           "format ascii 1.0\n"
           "comment written by VITAL\n"
           "element vertex " << landmarks->size() << "\n"
                                                     "property float x\n"
                                                     "property float y\n"
                                                     "property float z\n"
                                                     "property float nx\n"
                                                     "property float ny\n"
                                                     "property float nz\n"
                                                     "property uchar red\n"
                                                     "property uchar green\n"
                                                     "property uchar blue\n"
                                                     "property uint track_id\n"
                                                     "property uint observations\n"
                                                     "end_header\n";

  landmark_map::map_landmark_t lm_map = landmarks->landmarks();
  typedef  landmark_map::map_landmark_t::value_type lm_map_val_t;
  for( lm_map_val_t const& p : lm_map )
  {
    auto const& loc = p.second->loc();
    auto const& normal = p.second->normal();
    auto const& color = p.second->color();

    // the '+' prefix on the color values causes them to be printed
    // as decimal numbers instead of ASCII characters
    ofile << loc.x() << " " << loc.y() << " " << loc.z()
          << " " << normal.x() << " " << normal.y() << " " << normal.z()
          << " " << +color.r << " " << +color.g << " " << +color.b
          << " " << p.first << " " << p.second->observations() << "\n";
  }
  ofile.close();
} // write_ply_file

namespace {

// enumeration of the vertex properties we can handle
enum vertex_property_t
{
  INVALID,
  VX, VY, VZ,
  NX, NY, NZ,
  CR, CG, CB,
  INDEX,
  OBSERVATIONS,
};

/// Split a string into tokens delimited by whitespace
std::vector<std::string>
get_tokens(std::string const& line)
{
  std::istringstream iss( line );
  std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)),
                                  std::istream_iterator<std::string>());
  return tokens;
}

} // end anonymous namespace

/// Load a given \c landmark_map object from the specified PLY file path
landmark_map_sptr
read_ply_file( path_t const& file_path )
{
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    VITAL_THROW( file_not_found_exception, file_path, "Cannot find file." );
  }

  landmark_map::map_landmark_t landmarks;

  // open input file and read the tracks
  std::ifstream ifile( file_path.c_str() );

  if ( ! ifile )
  {
    VITAL_THROW( file_not_read_exception, file_path, "Cannot read file." );
  }

  // mapping between PLY vertex property names and our enum
  std::map<std::string, vertex_property_t> prop_map;
  // "standard" attributes
  prop_map["x"] = VX;
  prop_map["y"] = VY;
  prop_map["z"] = VZ;
  prop_map["nx"] = NX;
  prop_map["ny"] = NY;
  prop_map["nz"] = NZ;
  prop_map["red"] = CR;
  prop_map["green"] = CG;
  prop_map["blue"] = CB;
  // attributes defined by Vital
  prop_map["track_id"] = INDEX;
  prop_map["observations"] = OBSERVATIONS;
  // attributes for VisualSFM compatibility
  prop_map["vsfm_cnx"] = NX;
  prop_map["vsfm_cny"] = NY;
  prop_map["vsfm_cnz"] = NZ;
  prop_map["diffuse_red"] = CR;
  prop_map["diffuse_green"] = CG;
  prop_map["diffuse_blue"] = CB;
  prop_map["number_of_camera_sees_this_point"] = OBSERVATIONS;

  bool parsed_header = false;
  bool parsing_vertex_props = false;
  std::vector<vertex_property_t> vert_props;
  std::string line;

  unsigned int num_verts = 0, vert_count = 0;
  while ( std::getline( ifile, line ) )
  {
    std::vector<std::string> tokens = get_tokens(line);
    if ( line.empty() || tokens.empty() )
    {
      continue;
    }
    if ( ! parsed_header )
    {
      if ( line == "end_header" )
      {
        parsed_header = true;
        // TODO check that provided properties are meaningful
        // (e.g. has X, Y, and Z; has R, G, and B or no color, etc.)
        continue;
      }

      if ( tokens.size() == 3 &&
           tokens[0] == "element" &&
           tokens[1] == "vertex" )
      {
        std::istringstream iss(tokens[2]);
        iss >> num_verts;
        parsing_vertex_props = true;
      }
      else if ( tokens[0] == "element" )
      {
        parsing_vertex_props = false;
      }

      if ( parsing_vertex_props )
      {
        if ( tokens.size() == 3 && tokens[0] == "property" )
        {
          // map property names into enum values if supported
          std::string name = tokens[2];
          std::transform(name.begin(), name.end(), name.begin(), ::tolower);
          vertex_property_t prop = INVALID;
          const auto p = prop_map.find(name);
          if ( p != prop_map.end() )
          {
            prop = p->second;
          }
          vert_props.push_back(prop);
        }
      }

      continue;
    }

    // TODO throw exceptions if tokens.size() != vert_props.size()
    // or if the values do not parse as expected
    double x=0, y=0, z=0;
    double nx=0, ny=0, nz=0;
    rgb_color color;
    int cvalue;
    landmark_id_t id = static_cast<landmark_id_t>(vert_count++);
    unsigned observations = 0;
    for( unsigned int i=0; i<tokens.size() && i < vert_props.size(); ++i )
    {
      std::istringstream iss(tokens[i]);
      switch( vert_props[i] )
      {
        case VX:
          iss >> x;
          break;
        case VY:
          iss >> y;
          break;
        case VZ:
          iss >> z;
          break;
        case NX:
          iss >> nx;
          break;
        case NY:
          iss >> ny;
          break;
        case NZ:
          iss >> nz;
          break;
        case CR:
          iss >> cvalue;
          color.r = static_cast<unsigned char>(cvalue);
          break;
        case CG:
          iss >> cvalue;
          color.g = static_cast<unsigned char>(cvalue);
          break;
        case CB:
          iss >> cvalue;
          color.b = static_cast<unsigned char>(cvalue);
          break;
        case INDEX:
          iss >> id;
          break;
        case OBSERVATIONS:
          iss >> observations;
          break;
        case INVALID:
        default:
          break;
      }
    }

    std::shared_ptr<landmark_d> lm =
        std::make_shared<landmark_d>( vector_3d( x, y, z ) );
    lm->set_normal( { nx, ny, nz } );
    lm->set_color( color );
    lm->set_observations( observations );
    landmarks[id] = lm;

    // exit if we have read the expected number of points
    if ( vert_count > num_verts )
    {
      break;
    }
  }

  ifile.close();

  return landmark_map_sptr( new simple_landmark_map( landmarks ) );
} // read_ply_file

} } // end namespace
