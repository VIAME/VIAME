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
 * \brief Implementation of file IO functions for a \ref kwiver::vital::track_set
 *
 * \todo Describe format here.
 */

#include "track_set_io.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>

#include <vital/exceptions.h>
#include <vital/vital_foreach.h>
#include <vital/types/feature.h>
#include <vital/types/descriptor.h>
#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace vital {


/// Read in a track file, producing a track_set
track_set_sptr
read_track_file( path_t const& file_path )
{
  // Check that file exists
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    throw file_not_found_exception( file_path, "File does not exist." );
  }
  else if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw file_not_found_exception( file_path,
         "Path given doesn't point to a regular file!" );
  }

  // Reading in input file data
  std::ifstream input_stream( file_path.c_str(), std::fstream::in );
  if ( ! input_stream )
  {
    throw file_not_read_exception( file_path,
          "Could not open file at given path." );
  }

  // Read the file
  std::vector< track_sptr > tracks;
  std::map< track_id_t, track_sptr > track_map;
  for ( std::string line; std::getline( input_stream, line ); )
  {
    track_id_t tid;
    frame_id_t fid;
    feature_d feat;
    std::stringstream ss( line );
    ss >> tid >> fid;

    track_sptr t;
    std::map< track_id_t, track_sptr >::const_iterator it = track_map.find( tid );
    if ( it == track_map.end() )
    {
      t = track::make();
      t->set_id( tid );
      tracks.push_back( t );
      track_map[tid] = t;
    }
    else
    {
      t = it->second;
    }
    t->append( std::make_shared<track_state>( fid ) );
  }

  return track_set_sptr( new track_set( tracks ) );
} // read_track_file


/// Output the given \c track_set object to the specified file path
void
write_track_file( track_set_sptr const& tracks,
                  path_t const&         file_path )
{
  // If the track set is empty, throw
  if ( ! tracks || ( tracks->size() == 0 ) )
  {
    throw file_write_exception( file_path,
          "No tracks in the given track_set!" );
  }

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw file_write_exception( file_path,
          "Path given is a directory, can not write file." );
  }

  // Check that the directory of the given filepath exists, creating necessary
  // directories where needed.
  std::string parent_dir = kwiversys::SystemTools::GetFilenamePath(
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
  std::vector< vital::track_sptr > trks = tracks->tracks();
  VITAL_FOREACH( vital::track_sptr t, trks )
  {
    for( auto const& s : *t )
    {
      ofile << t->id() << " " << s->frame() << "\n";
    }
  }
  ofile.close();
} // write_track_file


/// Read in a track file, producing a track_set
feature_track_set_sptr
read_feature_track_file( path_t const& file_path )
{
  // Check that file exists
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    throw file_not_found_exception( file_path, "File does not exist." );
  }
  else if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw file_not_found_exception( file_path,
         "Path given doesn't point to a regular file!" );
  }

  // Reading in input file data
  std::ifstream input_stream( file_path.c_str(), std::fstream::in );
  if ( ! input_stream )
  {
    throw file_not_read_exception( file_path,
          "Could not open file at given path." );
  }

  // Read the file
  std::vector< track_sptr > tracks;
  std::map< track_id_t, track_sptr > track_map;
  for ( std::string line; std::getline( input_stream, line ); )
  {
    track_id_t tid;
    frame_id_t fid;
    auto feat = std::make_shared<feature_d>();
    std::stringstream ss( line );
    ss >> tid >> fid >> *feat;

    track_sptr t;
    std::map< track_id_t, track_sptr >::const_iterator it = track_map.find( tid );
    if ( it == track_map.end() )
    {
      t = track::make();
      t->set_id( tid );
      tracks.push_back( t );
      track_map[tid] = t;
    }
    else
    {
      t = it->second;
    }
    auto ftsd = std::make_shared<feature_track_state>(fid);
    ftsd->feature = feat;
    t->append( ftsd );
  }

  return std::make_shared<feature_track_set>( tracks );
} // read_track_file


/// Output the given \c track_set object to the specified file path
void
write_feature_track_file( feature_track_set_sptr const& tracks,
                          path_t const&                 file_path )
{
  // If the track set is empty, throw
  if ( ! tracks || ( tracks->size() == 0 ) )
  {
    throw file_write_exception( file_path,
          "No tracks in the given track_set!" );
  }

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    throw file_write_exception( file_path,
          "Path given is a directory, can not write file." );
  }

  // Check that the directory of the given filepath exists, creating necessary
  // directories where needed.
  std::string parent_dir = kwiversys::SystemTools::GetFilenamePath(
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
  std::vector< vital::track_sptr > trks = tracks->tracks();
  for( vital::track_sptr t : trks )
  {
    for( auto const& s : *t )
    {
      auto ftsd = std::dynamic_pointer_cast<feature_track_state>(s);
      if( !ftsd || !ftsd->feature )
      {
        throw invalid_data( "Provided track doest not contain a valid feature" );
      }
      ofile << t->id() << " " << s->frame() << " " << *ftsd->feature << "\n";
    }
  }
  ofile.close();
} // write_track_file


} } // end namespace
