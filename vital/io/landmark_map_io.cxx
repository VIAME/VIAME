/*ckwg +29
 * Copyright 2014 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Implementation of file IO functions for a \ref vital::landmark_map
 *
 * Uses the PLY file format
 */

#include "landmark_map_io.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include <vital/exceptions.h>
#include <vital/vital_foreach.h>
#include <kwiversys/SystemTools.hxx>

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
    throw file_write_exception( file_path,
         "No landmarks in the given landmark map!" );
  }

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw file_write_exception( file_path,
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
      throw file_write_exception( parent_dir,
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
                                                     "property uint track_id\n"
                                                     "end_header\n";

  landmark_map::map_landmark_t lm_map = landmarks->landmarks();
  typedef  landmark_map::map_landmark_t::value_type lm_map_val_t;
  VITAL_FOREACH( lm_map_val_t const& p, lm_map )
  {
    vector_3d loc = p.second->loc();

    ofile << loc.x() << " " << loc.y() << " " << loc.z()
          << " " << p.first << "\n";
  }
  ofile.close();
} // write_ply_file


/// Load a given \c landmark_map object from the specified PLY file path
landmark_map_sptr
read_ply_file( path_t const& file_path )
{
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    throw file_not_found_exception( file_path, "Cannot find file." );
  }

  landmark_map::map_landmark_t landmarks;

  // open input file and read the tracks
  std::ifstream ifile( file_path.c_str() );

  if ( ! ifile )
  {
    throw file_not_read_exception( file_path, "Cannot read file." );
  }

  bool parsed_header = false;
  std::string line;

  while ( std::getline( ifile, line ) )
  {
    if ( ! parsed_header || line.empty() )
    {
      if ( line == "end_header" )
      {
        parsed_header = true;
      }
      continue;
    }

    std::istringstream iss( line );

    double x, y, z;
    landmark_id_t id;

    iss >> x >> y >> z >> id;

    landmarks[id] = landmark_sptr( new landmark_d( vector_3d( x, y, z ) ) );
  }

  ifile.close();

  return landmark_map_sptr( new simple_landmark_map( landmarks ) );
} // read_ply_file

} } // end namespace
