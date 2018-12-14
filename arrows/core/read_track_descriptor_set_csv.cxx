/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief Implementation of read_track_descriptor_set_csv
 */

#include "read_track_descriptor_set_csv.h"

#include <vital/util/tokenize.h>
#include <vital/util/data_stream_reader.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace core {

// -------------------------------------------------------------------------------
class read_track_descriptor_set_csv::priv
{
public:
  priv( read_track_descriptor_set_csv* parent)
    : m_parent( parent )
    , m_logger( kwiver::vital::get_logger( "read_track_descriptor_set_csv" ) )
    , m_first( true )
    , m_batch_load( true )
    , m_read_raw_descriptor( true )
    , m_delim( "," )
    , m_sub_delim( " " )
    , m_current_idx( 0 )
    , m_last_idx( 1 )
  {}

  ~priv() {}

  read_track_descriptor_set_csv* m_parent;
  kwiver::vital::logger_handle_t m_logger;

  bool m_first;
  bool m_batch_load;

  bool m_read_raw_descriptor;

  std::string m_delim;
  std::string m_sub_delim;

  int m_current_idx;
  int m_last_idx;

  void read_all();

  // Map of track descriptors indexed by frame number. Each set contains all descs
  // referenced (active) on that individual frame.
  std::map< int, std::vector< vital::track_descriptor_sptr > > m_descs_by_frame_id;

  // Compilation of all loaded descriptors, track id -> track sptr mapping
  std::vector< vital::track_descriptor_sptr > m_all_descs;
};


// ===============================================================================
read_track_descriptor_set_csv
::read_track_descriptor_set_csv()
  : d( new read_track_descriptor_set_csv::priv( this ) )
{
}


read_track_descriptor_set_csv
::~read_track_descriptor_set_csv()
{
}


// -------------------------------------------------------------------------------
void
read_track_descriptor_set_csv
::open( std::string const& filename )
{
  vital::algo::read_track_descriptor_set::open( filename );

  d->m_first = true;

  d->m_descs_by_frame_id.clear();
  d->m_all_descs.clear();
}


// -------------------------------------------------------------------------------
void
read_track_descriptor_set_csv
::set_configuration(vital::config_block_sptr config)
{
  d->m_batch_load =
    config->get_value<bool>( "batch_load", d->m_batch_load );
  d->m_read_raw_descriptor =
    config->get_value<bool>( "read_raw_descriptor", d->m_batch_load );
}


// -------------------------------------------------------------------------------
bool
read_track_descriptor_set_csv
::check_configuration( vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------
bool
read_track_descriptor_set_csv
::read_set( kwiver::vital::track_descriptor_set_sptr& set )
{
  if( d->m_first )
  {
    // Read in all detections
    d->read_all();
    d->m_first = false;
  }

  if( d->m_batch_load )
  {
    set = std::make_shared< vital::track_descriptor_set >( d->m_all_descs );
    return true;
  }

  // Return detection set at current index if there is one
  if( d->m_descs_by_frame_id.count( d->m_current_idx ) == 0 )
  {
    // Return empty set
    set = std::make_shared< vital::track_descriptor_set >();
  }
  else
  {
    // Return descs for this frame
    vital::track_descriptor_set_sptr new_set(
      new vital::track_descriptor_set(
        d->m_descs_by_frame_id[ d->m_current_idx ] ) );

    set = new_set;
  }

  ++d->m_current_idx;

  // Return if we are done parsing
  return this->at_eof();
}


// -------------------------------------------------------------------------------
void
read_track_descriptor_set_csv::priv
::read_all()
{
  std::string line;
  vital::data_stream_reader stream_reader( m_parent->stream() );

  m_descs_by_frame_id.clear();
  m_all_descs.clear();

  while( stream_reader.getline( line ) )
  {
    if( !line.empty() && line[0] == '#' )
    {
      continue;
    }

    std::vector< std::string > tokens;
    vital::tokenize( line, tokens, m_delim, true );

    if( !tokens.empty() && tokens.size() != 8 )
    {
      std::stringstream str;

      str << "This is not a correct descriptor csv; found "
          << tokens.size() << " columns in\n\"" << line << "\"";

      throw vital::invalid_data( str.str() );
    }

    vital::track_descriptor_sptr desc = vital::track_descriptor::create( tokens[1] );

    desc->set_uid( vital::uid( tokens[0] ) );

    std::vector< std::string > tid_tokens, raw_tokens, hist_tokens;

    vital::tokenize( tokens[3], tid_tokens, m_sub_delim, true );
    vital::tokenize( tokens[7], hist_tokens, m_sub_delim, true );

    if( m_read_raw_descriptor )
    {
      vital::tokenize( tokens[5], raw_tokens, m_sub_delim, true );
    }

    unsigned tid_size = std::stoi( tokens[2] );
    unsigned desc_size = std::stoi( tokens[4] );
    unsigned hist_size = std::stoi( tokens[6] );

    bool contains_world_info = ( hist_size == hist_tokens.size() / 10 );

    if( tid_size != tid_tokens.size() ||
        ( m_read_raw_descriptor && desc_size != raw_tokens.size() ) ||
        ( !contains_world_info && hist_size != hist_tokens.size() / 6 ) )
    {
      throw vital::invalid_data( "Track descriptor reading size checksum failed" );
    }

    for( auto id : tid_tokens )
    {
      desc->add_track_id( std::stoi( id ) );
    }

    if( m_read_raw_descriptor )
    {
      desc->resize_descriptor( desc_size );

      for( unsigned i = 0; i < desc_size; ++i )
      {
        desc->at( i ) = std::stod( raw_tokens[i] );
      }
    }

    for( unsigned i = 0; i < hist_size; ++i )
    {
      unsigned start_ind = i * ( contains_world_info ? 10 : 6 );

      vital::timestamp ts(
        std::stoi( hist_tokens[ start_ind + 1 ] ),
        std::stoi( hist_tokens[ start_ind + 0 ] ) );

      vital::bounding_box_d bbox(
        std::stof( hist_tokens[ start_ind + 2 ] ),
        std::stof( hist_tokens[ start_ind + 3 ] ),
        std::stof( hist_tokens[ start_ind + 4 ] ),
        std::stof( hist_tokens[ start_ind + 5 ] ) );

      if( contains_world_info )
      {
        vital::bounding_box_d wrld_bbox(
          std::stof( hist_tokens[ start_ind + 6 ] ),
          std::stof( hist_tokens[ start_ind + 7 ] ),
          std::stof( hist_tokens[ start_ind + 8 ] ),
          std::stof( hist_tokens[ start_ind + 9 ] ) );

        desc->add_history_entry(
          vital::track_descriptor::history_entry(
            ts, bbox, wrld_bbox ) );
      }
      else
      {
        desc->add_history_entry(
          vital::track_descriptor::history_entry(
            ts, bbox ) );
      }
    }

    // Add track to indexes
    if( !m_batch_load && !desc->get_history().empty() )
    {
      int frame_index = desc->get_history().back().get_timestamp().get_frame();
      m_descs_by_frame_id[ frame_index ].push_back( desc );
      m_last_idx = std::max( m_last_idx, frame_index );
    }

    m_all_descs.push_back( desc );
  }
}

} } } // end namespace
