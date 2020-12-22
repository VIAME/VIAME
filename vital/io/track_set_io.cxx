// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
#include <vital/types/feature.h>
#include <vital/types/descriptor.h>
#include <kwiversys/SystemTools.hxx>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
/// Read in a track file, producing a track_set
track_set_sptr
read_track_file( path_t const& file_path )
{
  // Check that file exists
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    VITAL_THROW( file_not_found_exception,
                 file_path, "File does not exist." );
  }
  else if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_not_found_exception, file_path,
                 "Path given doesn't point to a regular file!" );
  }

  // Reading in input file data
  std::ifstream input_stream( file_path.c_str(), std::fstream::in );
  if ( ! input_stream )
  {
    VITAL_THROW( file_not_read_exception, file_path,
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
      t = track::create();
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

// ----------------------------------------------------------------------------
/// Output the given \c track_set object to the specified file path
void
write_track_file( track_set_sptr const& tracks,
                  path_t const&         file_path )
{
  // If the track set is empty, throw
  if ( ! tracks || ( tracks->size() == 0 ) )
  {
    VITAL_THROW( file_write_exception, file_path,
                 "No tracks in the given track_set!" );
  }

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_write_exception, file_path,
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
      VITAL_THROW( file_write_exception, parent_dir,
                   "Attempted directory creation, but no directory created! "
                   "No idea what happened here..." );
    }
  }

  // open output file and write the tracks
  std::ofstream ofile( file_path.c_str() );
  std::vector< vital::track_sptr > trks = tracks->tracks();
  for( vital::track_sptr t : trks )
  {
    for( auto const& s : *t )
    {
      ofile << t->id() << " " << s->frame() << "\n";
    }
  }
  ofile.close();
} // write_track_file

// ----------------------------------------------------------------------------
/// Read in a track file, producing a track_set
feature_track_set_sptr
read_feature_track_file( path_t const& file_path )
{
  // Check that file exists
  if ( ! kwiversys::SystemTools::FileExists( file_path ) )
  {
    VITAL_THROW( file_not_found_exception,
                 file_path, "File does not exist." );
  }
  else if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_not_found_exception, file_path,
                 "Path given doesn't point to a regular file!" );
  }

  // Reading in input file data
  std::ifstream input_stream( file_path.c_str(), std::fstream::in );
  if ( ! input_stream )
  {
    VITAL_THROW( file_not_read_exception, file_path,
                 "Could not open file at given path." );
  }

  std::set<frame_id_t> frames_in_track_set;
  // Read the file
  std::vector< track_sptr > tracks;
  std::map< track_id_t, track_sptr > track_map;
  for ( std::string line; std::getline( input_stream, line ); )
  {
    track_id_t tid;
    frame_id_t fid;
    auto feat = std::make_shared<feature_d>();
    std::stringstream ss( line );
    if (ss.str() == "keyframes")
    {
      break;
    }

    bool has_desc = false;

    ss >> tid >> fid >> *feat;
    if (!(ss >> has_desc))
    {
      //all older files without has_desc used only descriptor based features
      has_desc = true;
    }

    track_sptr t;
    std::map< track_id_t, track_sptr >::const_iterator it = track_map.find( tid );
    if ( it == track_map.end() )
    {
      t = track::create();
      t->set_id( tid );
      tracks.push_back( t );
      track_map[tid] = t;
    }
    else
    {
      t = it->second;
    }
    frames_in_track_set.insert(fid);

    auto ftsd = std::make_shared<feature_track_state>(fid);
    ftsd->feature = feat;
    if (has_desc)
    {
      ftsd->descriptor = std::make_shared<descriptor_fixed<unsigned char,1>>();  //dummy descriptor.
      //this will be overwritten when the descriptor file is read.
    }
    t->append( ftsd );
  }

  feature_track_set_sptr fts = std::make_shared<feature_track_set>( tracks );

  for (std::string line; std::getline(input_stream, line); )
  {
    frame_id_t fid;
    std::stringstream ss(line);
    ss >> fid;
    auto frame_data = std::make_shared<feature_track_set_frame_data>();
    frame_data->is_keyframe = true;
    fts->set_frame_data(frame_data, fid);
  }

  //create frame_data with is_keyframe set to false for all non-keyframes
  auto frame_ids = fts->all_frame_ids();
  for (auto fid : frame_ids)
  {
    if (!fts->frame_data(fid))
    {
      auto frame_data = std::make_shared<feature_track_set_frame_data>();
      frame_data->is_keyframe = false;
      fts->set_frame_data(frame_data, fid);
    }
  }

  return fts;
} // read_track_file

// ----------------------------------------------------------------------------
/// Output the given \c track_set object to the specified file path
void
write_feature_track_file( feature_track_set_sptr const& tracks,
                          path_t const&                 file_path )
{
  // If the track set is empty, throw
  if ( ! tracks || ( tracks->size() == 0 ) )
  {
    VITAL_THROW( file_write_exception, file_path,
                 "No tracks in the given track_set!" );
  }

  // If the given path is a directory, we obviously can't write to it.
  if ( kwiversys::SystemTools::FileIsDirectory( file_path ) )
  {
    VITAL_THROW( file_write_exception, file_path,
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
      VITAL_THROW( file_write_exception, parent_dir,
                   "Attempted directory creation, but no directory created! "
                   "No idea what happened here..." );
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
        VITAL_THROW( invalid_data, "Provided track doest not contain a valid feature" );
      }
      bool has_desc = ftsd->descriptor.get() != NULL;
      ofile << t->id() << " " << s->frame() << " " << *ftsd->feature << " " << has_desc << "\n";
    }
  }

  std::set<frame_id_t> keyframes = tracks->keyframes();
  if ( !keyframes.empty() )
  {
    ofile << "keyframes" << "\n";
    for (auto k : keyframes)
    {
      ofile << k << "\n";
    }
  }
  ofile.close();
} // write_track_file

} } // end namespace
