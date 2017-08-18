/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include "perform_query_process.h"

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

#include <vital/vital_foreach.h>

namespace kwiver
{

//------------------------------------------------------------------------------
// Private implementation class
class perform_query_process::priv
{
public:
  priv();
  ~priv();
}; // end priv class


// =============================================================================

perform_query_process
::perform_query_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new perform_query_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}


perform_query_process
::~perform_query_process()
{
}


// -----------------------------------------------------------------------------
void perform_query_process
::_configure()
{
}


// -----------------------------------------------------------------------------
void
perform_query_process
::_step()
{
  // Retrieve inputs from ports
  vital::database_query_sptr query;
  vital::iqr_feedback_sptr feedback;

  query = grab_from_port_using_trait( database_query );
  feedback = grab_from_port_using_trait( iqr_feedback );

  vital::query_result_set_sptr output( new vital::query_result_set() );

  for( unsigned i = 1; i < 4; i++ )
  {
    vital::query_result_sptr entry( new vital::query_result() );

    vital::timestamp ts1( 1375007280949983, 7 );
    vital::timestamp ts2( 1375007281050083, 8 );
    vital::timestamp ts3( 1375007281150183, 9 );

    vital::timestamp ts4( 1375007477946783, 1975 );
    vital::timestamp ts5( 1375007478046883, 1976 );
    vital::timestamp ts6( 1375007478146883, 1977 );

    entry->set_stream_id( "/data/virat/video/aphill/09172008flight1tape1_5.mpg" );
    entry->set_instance_id( i );
    entry->set_relevancy_score( ( 4 - i ) * 0.30 );

    typedef vital::track_descriptor td;

    td::descriptor_data_sptr_t data( new td::descriptor_data_t( 100 ) );

    for( unsigned i = 0; i < 100; i++ )
    {
      (data->raw_data())[i] = static_cast<double>( i );
    }

    td::history_entry::image_bbox_t region1( 40, 40, 100, 100 );
    td::history_entry::image_bbox_t region2( 140, 140, 200, 200 );

    td::history_entry hist_entry1( ts1, region1 );
    td::history_entry hist_entry2( ts2, region1 );
    td::history_entry hist_entry3( ts3, region1 );

    td::history_entry hist_entry4( ts4, region2 );
    td::history_entry hist_entry5( ts5, region2 );
    td::history_entry hist_entry6( ts6, region2 );

    if( i == 1 )
    {
      vital::track_descriptor_set_sptr desc_set( new vital::track_descriptor_set() );
      vital::track_descriptor_sptr new_desc = td::create( "cnn_descriptor" );

      new_desc->set_descriptor( data );

      new_desc->add_history_entry( hist_entry1 );
      new_desc->add_history_entry( hist_entry2 );
      new_desc->add_history_entry( hist_entry3 );

      desc_set->push_back( new_desc );
      entry->set_descriptors( desc_set );
      entry->set_temporal_bounds( ts1, ts3 );
    }
    else if( i == 2 )
    {
      vital::track_descriptor_set_sptr desc_set( new vital::track_descriptor_set() );
      vital::track_descriptor_sptr new_desc = td::create( "cnn_descriptor" );

      new_desc->set_descriptor( data );

      new_desc->add_history_entry( hist_entry4 );
      new_desc->add_history_entry( hist_entry5 );
      new_desc->add_history_entry( hist_entry6 );

      desc_set->push_back( new_desc );

      entry->set_descriptors( desc_set );
      entry->set_temporal_bounds( ts4, ts6 );

      vital::track_sptr trk = vital::track::create();

      vital::detected_object_sptr det1(
        new vital::detected_object( region2 ) );
      vital::detected_object_sptr det2(
        new vital::detected_object( region2 ) );
      vital::detected_object_sptr det3(
        new vital::detected_object( region2 ) );

      vital::track_state_sptr state1(
        new vital::object_track_state( ts4.get_frame(), det1 ) );
      vital::track_state_sptr state2(
        new vital::object_track_state( ts5.get_frame(), det2 ) );
      vital::track_state_sptr state3(
        new vital::object_track_state( ts6.get_frame(), det3 ) );

      trk->set_id( 13 );

      trk->append( state1 );
      trk->append( state2 );
      trk->append( state3 );

      std::vector< vital::track_sptr > trk_vec;
      trk_vec.push_back( trk );
    
      vital::object_track_set_sptr trk_set(
        new vital::object_track_set( trk_vec ) );

      new_desc->add_track_id( 13 );

      entry->set_tracks( trk_set );
    }
    else if( i == 3 )
    {
      entry->set_temporal_bounds( ts1, ts6 );
    }

    output->push_back( entry );
  }

  if( feedback && ( !feedback->positive_ids().empty() || !feedback->negative_ids().empty() ) )
  {
    std::reverse( output->begin(), output->end() );

    double count = 0.90 + ( 0.1 * (double)rand() / (double)RAND_MAX );

    VITAL_FOREACH( auto item, *output )
    {
      item->set_relevancy_score( count );
      count -= ( 0.4 * (double)rand() / (double)RAND_MAX );
    }
  }

  push_to_port_using_trait( query_result, output );
}


// -----------------------------------------------------------------------------
void perform_query_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( database_query, required );
  declare_input_port_using_trait( iqr_feedback, optional );

  // -- output --
  declare_output_port_using_trait( query_result, optional );
}


// -----------------------------------------------------------------------------
void perform_query_process
::make_config()
{
}


// =============================================================================
perform_query_process::priv
::priv()
{
}


perform_query_process::priv
::~priv()
{
}

} // end namespace
