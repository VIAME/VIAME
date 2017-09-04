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

#include <vital/algo/read_object_track_set.h>
#include <vital/algo/read_track_descriptor_set.h>

#include <boost/filesystem.hpp>

#include <tuple>

namespace kwiver
{

#define DUMMY_OUTPUT 1

namespace algo = vital::algo;

create_port_trait( external_descriptor_set, descriptor_set,
  "Descriptor set to be processed by external query handler" );
create_port_trait( external_exemplar_uids, string_vector,
  "Descriptor set UIDs to be processed by external query handler" );
create_port_trait( external_positive_uids, string_vector,
  "Descriptor set positive IQR exemplars" );
create_port_trait( external_negative_uids, string_vector,
  "Descriptor set negative IQR exemplars" );
create_port_trait( result_descriptor_uids, string_vector,
  "Descriptor set response from external query handler" );
create_port_trait( result_descriptor_scores, double_vector,
  "Descriptor set scores from external query handler" );

create_config_trait( external_handler, bool,
  "true", "Whether or not an external query handler is used" );
create_config_trait( database_folder, std::string,
  "", "Folder containing all track and descriptor files" );
create_config_trait( max_result_count, unsigned,
  "100", "Maximum number of results to return at once" );
create_config_trait( track_postfix, std::string,
  "_tracks.kw18", "Postfix to add to basename for track files" );
create_config_trait( descriptor_postfix, std::string,
  "_descriptors.csv", "Postfix to add to basename for desc files" );
create_config_trait( index_postfix, std::string,
  ".index", "Postfix to add to basename for reading index files" );

//------------------------------------------------------------------------------
// Private implementation class
class perform_query_process::priv
{
public:
  explicit priv( perform_query_process* p );
  ~priv();

  perform_query_process* parent;

  bool external_handler;
  std::string database_folder;

  std::string track_postfix;
  std::string descriptor_postfix;
  std::string index_postfix;

  unsigned max_result_count;

  bool is_first;
  std::map< unsigned, vital::query_result_sptr > previous_results;
  std::map< std::string, unsigned > instance_ids;
  std::map< unsigned, vital::query_result_sptr > forced_positives;
  std::map< unsigned, vital::query_result_sptr > forced_negatives;
  vital::uid active_uid;

  unsigned result_counter;
  bool database_populated;

  algo::read_track_descriptor_set_sptr descriptor_reader;
  algo::read_object_track_set_sptr track_reader;

  // Video name <=> descriptor sptr <=> track sptr tuple
  typedef std::tuple< std::string,
                      vital::track_descriptor_sptr,
                      std::vector< vital::track_sptr > > desc_tuple_t;

  std::map< std::string, desc_tuple_t > uid_to_desc;

  void populate_database();

  void reset_query( const vital::database_query_sptr& query );
  unsigned get_instance_id( const std::string& uid );
}; // end priv class


// =============================================================================

perform_query_process
::perform_query_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new perform_query_process::priv( this ) )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  // Required for external feedback loop
  set_data_checking_level( check_none );

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
  vital::config_block_sptr algo_config = get_config();

  d->external_handler = config_value_using_trait( external_handler );
  d->database_folder = config_value_using_trait( database_folder );
  d->max_result_count = config_value_using_trait( max_result_count );
  d->track_postfix = config_value_using_trait( track_postfix );
  d->descriptor_postfix = config_value_using_trait( descriptor_postfix );
  d->index_postfix = config_value_using_trait( index_postfix );

  if( d->external_handler )
  {
    algo::read_track_descriptor_set::set_nested_algo_configuration(
      "descriptor_reader", algo_config, d->descriptor_reader );

    if( !d->descriptor_reader )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Unable to create descriptor reader" );
    }

    algo::read_track_descriptor_set::get_nested_algo_configuration(
      "descriptor_reader", algo_config, d->descriptor_reader );

    if( !algo::read_track_descriptor_set::check_nested_algo_configuration(
      "descriptor_reader", algo_config ) )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Configuration check failed." );
    }

    algo::read_object_track_set::set_nested_algo_configuration(
      "track_reader", algo_config, d->track_reader );

    if( !d->track_reader )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Unable to create track reader" );
    }

    algo::read_object_track_set::get_nested_algo_configuration(
      "track_reader", algo_config, d->track_reader );

    if( !algo::read_object_track_set::check_nested_algo_configuration(
      "track_reader", algo_config ) )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Configuration check failed." );
    }
  }
}


// -----------------------------------------------------------------------------
void
perform_query_process
::_step()
{
  // Check for termination since we are in manual mode
  auto port_info = peek_at_port_using_trait( database_query );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( database_query );
    grab_edge_datum_using_trait( iqr_feedback );
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( query_result, dat );

    if( d->external_handler )
    {
      push_datum_to_port_using_trait( external_descriptor_set, dat );
      push_datum_to_port_using_trait( external_exemplar_uids, dat );
      push_datum_to_port_using_trait( external_positive_uids, dat );
      push_datum_to_port_using_trait( external_negative_uids, dat );
    }

    return;
  }

  // Retrieve inputs from ports
  vital::database_query_sptr query;
  vital::iqr_feedback_sptr feedback;

  query = grab_from_port_using_trait( database_query );
  feedback = grab_from_port_using_trait( iqr_feedback );

  // Declare output
  vital::query_result_set_sptr output( new vital::query_result_set() );

  // No query received, do nothing, return no results
  if( !query && !feedback )
  {
    push_to_port_using_trait( query_result, output );
    return;
  }

  // Reset query when no IQR information is provided
  if( d->is_first || !feedback ||
    ( feedback->positive_ids().empty() &&
      feedback->negative_ids().empty() ) )
  {
    d->reset_query( query );
    d->is_first = false;
  }

  // Call external feedback loop if enabled
  if( d->external_handler )
  {
    d->populate_database();

    vital::string_vector_sptr positive_uids( new vital::string_vector() );
    vital::string_vector_sptr negative_uids( new vital::string_vector() );

    if( feedback &&
      ( !feedback->positive_ids().empty() ) )
    {
      VITAL_FOREACH( auto id, feedback->positive_ids() )
      {
        VITAL_FOREACH( auto desc_sptr, *d->previous_results[id]->descriptors() )
        {
          positive_uids->push_back( desc_sptr->get_uid().value() );
        }

        d->forced_positives[ id ] = d->previous_results[ id ];

        auto negative_itr = d->forced_negatives.find( id );

        if( negative_itr != d->forced_negatives.end() )
        {
          d->forced_negatives.erase( negative_itr );
        }
      }
    }

    if( feedback &&
      ( !feedback->negative_ids().empty() ) )
    {
      VITAL_FOREACH( auto id, feedback->negative_ids() )
      {
        VITAL_FOREACH( auto desc_sptr, *d->previous_results[id]->descriptors() )
        {
          negative_uids->push_back( desc_sptr->get_uid().value() );
        }

        d->forced_negatives[ id ] = d->previous_results[ id ];

        auto positive_itr = d->forced_positives.find( id );

        if( positive_itr != d->forced_positives.end() )
        {
          d->forced_positives.erase( positive_itr );
        }
      }
    }

#ifndef DUMMY_OUTPUT
    // Format data to simplified format for external
    std::vector< vital::descriptor_sptr > exemplar_raw_descs;

    vital::string_vector_sptr exemplar_uids( new vital::string_vector() );

    VITAL_FOREACH( auto track_desc, *query->descriptors() )
    {
      exemplar_uids->push_back( track_desc->get_uid().value() );
      exemplar_raw_descs.push_back( track_desc->get_descriptor() );
    }

    vital::descriptor_set_sptr exemplar_descs(
      new vital::simple_descriptor_set( exemplar_raw_descs ) );

    vital::string_vector_sptr result_uids;
    vital::double_vector_sptr result_scores;

    // Send data to external process
    push_to_port_using_trait( external_descriptor_set, exemplar_descs );
    push_to_port_using_trait( external_exemplar_uids, exemplar_uids );
    push_to_port_using_trait( external_positive_uids, positive_uids );
    push_to_port_using_trait( external_negative_uids, negative_uids );

    // Receive data from external process (halts until finished)
    result_uids = grab_from_port_using_trait( result_descriptor_uids );
    result_scores = grab_from_port_using_trait( result_descriptor_scores );
#else
    // Format data to simplified format for external
    vital::string_vector_sptr result_uids( new vital::string_vector() );
    vital::double_vector_sptr result_scores( new vital::double_vector() );

    result_uids->push_back( "output_frame_final_item_14" );
    result_uids->push_back( "output_frame_final_item_13" );
    result_uids->push_back( "output_frame_final_item_12" );
    result_uids->push_back( "output_frame_final_item_11" );
    result_uids->push_back( "output_frame_final_item_10" );
    result_uids->push_back( "output_frame_final_item_9" );
    result_uids->push_back( "output_frame_final_item_8" );
    result_uids->push_back( "output_frame_final_item_7" );
    result_uids->push_back( "output_frame_final_item_21" );
    result_uids->push_back( "output_frame_final_item_24" );

    result_scores->push_back( 0.98 );
    result_scores->push_back( 0.95 );
    result_scores->push_back( 0.92 );
    result_scores->push_back( 0.91 );
    result_scores->push_back( 0.90 );
    result_scores->push_back( 0.80 );
    result_scores->push_back( 0.79 );
    result_scores->push_back( 0.69 );
    result_scores->push_back( 0.68 );
    result_scores->push_back( 0.50 );
    result_scores->push_back( 0.45 );
#endif

    // Handle forced positive examples, set score to 1, make sure at front
    for( auto itr = d->forced_positives.begin();
         itr != d->forced_positives.end(); itr++ )
    {
      itr->second->set_relevancy_score( 1.0 );
      output->push_back( itr->second );
    }

    // Handle all new or unadjudacted results
    for( unsigned i = 0; i < result_uids->size(); ++i )
    {
      if( i > d->max_result_count )
      {
        break;
      }

      auto result_uid = (*result_uids)[i];
      auto result_score = (*result_scores)[i];

      auto db_res = d->uid_to_desc.find( result_uid );

      if( db_res == d->uid_to_desc.end() )
      {
        continue;
      }

      // Create result set and set relevant IDs
      auto iid = d->get_instance_id( result_uid );

      // Check if result is forced positive or negative (e.g. annotated by user)
      if( d->forced_positives.find( iid ) != d->forced_positives.end() ||
          d->forced_negatives.find( iid ) != d->forced_negatives.end() )
      {
        continue;
      }

      vital::query_result_sptr entry( new vital::query_result() );

      entry->set_query_id( d->active_uid );
      entry->set_stream_id( std::get<0>( db_res->second ) );
      entry->set_instance_id( iid );
      entry->set_relevancy_score( result_score );

      // Assign track descriptor set to result
      vital::track_descriptor_set_sptr desc_set(
        new vital::track_descriptor_set() );

      desc_set->push_back( std::get<1>( db_res->second ) );
      entry->set_descriptors( desc_set );

      // Assign temporal bounds to this query result
      vital::timestamp ts1, ts2;
      bool is_first = true;

      VITAL_FOREACH( auto desc, *desc_set )
      {
        VITAL_FOREACH( auto hist, desc->get_history() )
        {
          if( is_first )
          {
            ts1 = hist.get_timestamp();
            ts2 = hist.get_timestamp();

            is_first = false;
          }
          else if( hist.get_timestamp().get_frame() < ts1.get_frame() )
          {
            ts1 = hist.get_timestamp();
          }
          else if( hist.get_timestamp().get_frame() > ts2.get_frame() )
          {
            ts2 = hist.get_timestamp();
          }
        }
      }
      entry->set_temporal_bounds( ts1, ts2 );

      // Assign track set to result
      vital::object_track_set_sptr trk_set(
        new vital::object_track_set( std::get<2>( db_res->second ) ) );

      entry->set_tracks( trk_set );

      // Remember this descriptor result for future iterations
      d->previous_results[ entry->instance_id() ] = entry;

      output->push_back( entry );
    }

    // Handle forced negative examples, set score to 0, make sure at end of result set
    for( auto itr = d->forced_negatives.begin();
         itr != d->forced_negatives.end(); itr++ )
    {
      itr->second->set_relevancy_score( 0.0 );
      output->push_back( itr->second );
    }
  }
  else
  {
    throw std::runtime_error( "Only external handler mode yet supported" );
  }

  // Push outputs downstream
  push_to_port_using_trait( query_result, output );
}


// -----------------------------------------------------------------------------
void perform_query_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t optional_no_dep;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );
  optional_no_dep.insert( flag_input_nodep );

  // -- input --
  declare_input_port_using_trait( database_query, required );
  declare_input_port_using_trait( iqr_feedback, optional );

  // -- output --
  declare_output_port_using_trait( query_result, optional );

  // -- feedback loop --
  declare_output_port_using_trait( external_descriptor_set, optional );
  declare_output_port_using_trait( external_exemplar_uids, optional );
  declare_output_port_using_trait( external_positive_uids, optional );
  declare_output_port_using_trait( external_negative_uids, optional );

  declare_input_port_using_trait( result_descriptor_uids, optional_no_dep );
  declare_input_port_using_trait( result_descriptor_scores, optional_no_dep );
}


// -----------------------------------------------------------------------------
void perform_query_process
::make_config()
{
  declare_config_using_trait( external_handler );
  declare_config_using_trait( database_folder );
  declare_config_using_trait( max_result_count );
  declare_config_using_trait( descriptor_postfix );
  declare_config_using_trait( track_postfix );
  declare_config_using_trait( index_postfix );
}


// =============================================================================
perform_query_process::priv
::priv( perform_query_process* p )
 : parent( p )
 , external_handler( true )
 , database_folder( "" )
 , max_result_count( 100 )
 , is_first( true )
 , database_populated( false )
{
}


perform_query_process::priv
::~priv()
{
}


void perform_query_process::priv
::populate_database()
{
  if( database_populated )
  {
    return;
  }

  // List all files to check
  std::vector< std::string > basenames;

  boost::filesystem::path dir( database_folder );

  for( boost::filesystem::directory_iterator file_iter( dir );
       file_iter != boost::filesystem::directory_iterator();
       ++file_iter )
  {
    if( boost::filesystem::is_regular_file( *file_iter ) &&
        file_iter->path().extension().string() == index_postfix )
    {
      basenames.push_back( file_iter->path().stem().string() );
    }
  }

  // Load tracks for every base name
  VITAL_FOREACH( std::string name, basenames )
  {
    std::string track_file = database_folder + "/" + name + track_postfix;
    std::string desc_file = database_folder + "/" + name + descriptor_postfix;

    descriptor_reader->open( desc_file );
    track_reader->open( track_file );

    vital::track_descriptor_set_sptr descs;
    vital::object_track_set_sptr tracks;

    if( !descriptor_reader->read_set( descs ) )
    {
      LOG_ERROR( parent->logger(), "Unable to load desc set " << desc_file );
      continue;
    }

    if( !track_reader->read_set( tracks ) )
    {
      LOG_ERROR( parent->logger(), "Unable to load track set " << track_file );
      continue;
    }

    std::map< unsigned, vital::track_sptr > id_to_track;

    VITAL_FOREACH( auto trk_sptr, tracks->tracks() )
    {
      id_to_track[ trk_sptr->id() ] = trk_sptr;
    }

    VITAL_FOREACH( auto desc_sptr, *descs )
    {
      // Identify associated tracks
      std::vector< vital::track_sptr > assc_trks;

      VITAL_FOREACH( auto id, desc_sptr->get_track_ids() )
      {
        assc_trks.push_back( id_to_track[ id ] );
      }

      // Add to index
      uid_to_desc[ desc_sptr->get_uid().value() ] =
        desc_tuple_t( name, desc_sptr, assc_trks );
    }
  }

  database_populated = true;
}


void perform_query_process::priv
::reset_query( const vital::database_query_sptr& query )
{
  result_counter = 0;
  instance_ids.clear();
  previous_results.clear();
  forced_positives.clear();
  forced_negatives.clear();
  active_uid = query->id();
}


unsigned perform_query_process::priv
::get_instance_id( const std::string& uid )
{
  auto itr = instance_ids.find( uid );

  if( itr != instance_ids.end() )
  {
    return itr->second;
  }

  instance_ids[ uid ] = ++result_counter;
  return result_counter;
}

} // end namespace
