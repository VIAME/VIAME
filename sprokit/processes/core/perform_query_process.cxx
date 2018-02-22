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

#include <vital/algo/query_track_descriptor_set.h>

#include <sprokit/processes/adapters/embedded_pipeline.h>

#include <boost/filesystem.hpp>

#include <fstream>
#include <tuple>

namespace kwiver
{

namespace algo = vital::algo;

create_config_trait( external_handler, bool,
  "true", "Whether or not an external query handler is used" );
create_config_trait( external_pipeline_file, std::string,
  "", "External pipeline definition file location" );
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
create_config_trait( use_tracks_for_history, bool,
  "false", "Use object track states for track descriptor history" );
create_config_trait( merge_duplicate_results, bool,
  "false", "If use_tracks_for_history is on, use the track with the "
           "highest confidence, otherwise concatenate the history of "
           "all entries with the same track" );

//------------------------------------------------------------------------------
// Private implementation class
class perform_query_process::priv
{
public:
  explicit priv( perform_query_process* p );
  ~priv();

  perform_query_process* parent;

  bool external_handler;
  std::string external_pipeline_file;
  std::string database_folder;

  std::string track_postfix;
  std::string descriptor_postfix;
  std::string index_postfix;
  bool use_tracks_for_history;
  bool merge_duplicate_results;

  std::unique_ptr< embedded_pipeline > external_pipeline;

  unsigned max_result_count;

  bool is_first;
  std::map< unsigned, vital::query_result_sptr > previous_results;
  std::map< std::string, unsigned > instance_ids;
  std::map< unsigned, vital::query_result_sptr > forced_positives;
  std::map< unsigned, vital::query_result_sptr > forced_negatives;
  vital::uid active_uid;

  unsigned result_counter;
  bool database_populated;

  algo::query_track_descriptor_set_sptr descriptor_query;

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

  make_ports();
  make_config();
}


perform_query_process
::~perform_query_process()
{
  if( d->external_pipeline )
  {
    d->external_pipeline->send_end_of_input();
    d->external_pipeline->receive();
    d->external_pipeline->wait();
    d->external_pipeline.reset();
  }
}


// -----------------------------------------------------------------------------
void perform_query_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  d->external_handler = config_value_using_trait( external_handler );
  d->external_pipeline_file = config_value_using_trait( external_pipeline_file );
  d->database_folder = config_value_using_trait( database_folder );
  d->max_result_count = config_value_using_trait( max_result_count );
  d->track_postfix = config_value_using_trait( track_postfix );
  d->descriptor_postfix = config_value_using_trait( descriptor_postfix );
  d->index_postfix = config_value_using_trait( index_postfix );
  d->use_tracks_for_history = config_value_using_trait( use_tracks_for_history );
  d->merge_duplicate_results = config_value_using_trait( merge_duplicate_results );

  if( d->external_handler )
  {
    algo::query_track_descriptor_set::set_nested_algo_configuration(
      "descriptor_query", algo_config, d->descriptor_query );

    if( !d->descriptor_query )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Unable to create descriptor query" );
    }

    algo::query_track_descriptor_set::get_nested_algo_configuration(
      "descriptor_query", algo_config, d->descriptor_query );

    if( !algo::query_track_descriptor_set::check_nested_algo_configuration(
      "descriptor_query", algo_config ) )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Configuration check failed." );
    }

    d->descriptor_query->use_tracks_for_history( d->use_tracks_for_history );
  }
}


// -----------------------------------------------------------------------------
void
perform_query_process
::_init()
{
  auto dir = boost::filesystem::path( d->external_pipeline_file ).parent_path();

  if( d->external_handler && !d->external_pipeline_file.empty() )
  {
    std::unique_ptr< embedded_pipeline > new_pipeline =
      std::unique_ptr< embedded_pipeline >( new embedded_pipeline() );

    std::ifstream pipe_stream;
    pipe_stream.open( d->external_pipeline_file, std::ifstream::in );

    if( !pipe_stream )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Unable to open pipeline file: " + d->external_pipeline_file );
    }

    try
    {
      new_pipeline->build_pipeline( pipe_stream, dir.string() );
      new_pipeline->start();
    }
    catch( const std::exception& e )
    {
      throw sprokit::invalid_configuration_exception( name(), e.what() );
    }

    d->external_pipeline = std::move( new_pipeline );
    pipe_stream.close();
  }
}


// -----------------------------------------------------------------------------
static void
merge_history( vital::track_descriptor::descriptor_history_t& dest,
               vital::track_descriptor::descriptor_history_t const& src)
{
  auto dest_it = dest.begin();
  auto src_it = src.begin();

  while( true )
  {
    if( dest_it == dest.end() )
    {
      if( src_it == src.end() )
      {
        return;
      }
      else
      {
        dest.insert( dest_it, *src_it );
        src_it++;
        dest_it++;
      }
    }
    else if( src_it == src.end() || src_it->get_timestamp().get_frame() > dest_it->get_timestamp().get_frame() )
    {
      dest_it++;
    }
    else if( src_it->get_timestamp().get_frame() == dest_it->get_timestamp().get_frame() )
    {
      src_it++;
      dest_it++;
    }
    else
    {
      dest.insert( dest_it, *src_it );
      src_it++;
      dest_it++;
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
    return;
  }

  // Retrieve inputs from ports
  vital::database_query_sptr query;
  vital::iqr_feedback_sptr feedback;
  vital::uchar_vector_sptr model;

  query = grab_from_port_using_trait( database_query );
  feedback = grab_from_port_using_trait( iqr_feedback );
  model = grab_from_port_using_trait( iqr_model );

  // Declare output
  vital::query_result_set_sptr output( new vital::query_result_set() );

  // No query received, do nothing, return no results
  if( !query && !feedback && !model )
  {
    push_to_port_using_trait( query_result, output );
    push_to_port_using_trait( iqr_model, model );
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
    vital::string_vector_sptr positive_uids( new vital::string_vector() );
    vital::string_vector_sptr negative_uids( new vital::string_vector() );

    if( feedback &&
      ( !feedback->positive_ids().empty() ) )
    {
      for( auto id : feedback->positive_ids() )
      {
        for( auto desc_sptr : *d->previous_results[id]->descriptors() )
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
      for( auto id : feedback->negative_ids() )
      {
        for( auto desc_sptr : *d->previous_results[id]->descriptors() )
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

    if( query )
    {
      for( auto track_desc : *query->descriptors() )
      {
        exemplar_uids->push_back( track_desc->get_uid().value() );
        exemplar_raw_descs.push_back( track_desc->get_descriptor() );
      }
    }

    vital::descriptor_set_sptr exemplar_descs(
      new vital::simple_descriptor_set( exemplar_raw_descs ) );

    // Set request on pipeline inputs
    auto ids = adapter::adapter_data_set::create();

    ids->add_value( "descriptor_set", exemplar_descs );
    ids->add_value( "exemplar_uids", exemplar_uids );
    ids->add_value( "positive_uids", positive_uids );
    ids->add_value( "negative_uids", negative_uids );
    ids->add_value( "query_model", model );

    // Send the request through the pipeline and wait for a result
    d->external_pipeline->send( ids );

    auto const& ods = d->external_pipeline->receive();

    if( ods->is_end_of_data() )
    {
      throw std::runtime_error( "Pipeline terminated unexpectingly" );
    }

    // Grab result from pipeline output data set
    auto const& iter1 = ods->find( "result_uids" );
    auto const& iter2 = ods->find( "result_scores" );
    auto const& iter3 = ods->find( "result_model" );

    if( iter1 == ods->end() || iter2 == ods->end() || iter3 == ods->end() )
    {
      throw std::runtime_error( "Empty pipeline output" );
    }

    vital::string_vector_sptr result_uids =
      iter1->second->get_datum< vital::string_vector_sptr >();
    vital::double_vector_sptr result_scores =
      iter2->second->get_datum< vital::double_vector_sptr >();

    model = iter3->second->get_datum< vital::uchar_vector_sptr >();
#else
    // Format data to simplified format for external
    vital::string_vector_sptr result_uids( new vital::string_vector() );
    vital::double_vector_sptr result_scores( new vital::double_vector() );

    model = vital::uchar_vector_sptr( new vital::uchar_vector() );

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

    typedef std::pair< std::string, vital::track_id_t > unique_track_id_t;
    std::map< unique_track_id_t, vital::query_result_sptr > top_results;

    // Handle all new or unadjudacted results
    for( unsigned i = 0; i < result_uids->size(); ++i )
    {
      if( i > d->max_result_count )
      {
        break;
      }

      auto result_uid = (*result_uids)[i];
      auto result_score = (*result_scores)[i];

      vital::algo::query_track_descriptor_set::desc_tuple_t result;
      if( !d->descriptor_query->get_track_descriptor( result_uid, result ))
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

      vital::query_result_sptr entry;
      bool insert = true;

      // If there is more than one track for a descriptor, there's no point in
      // trying to do any merging
      if( d->merge_duplicate_results && std::get<2>( result ).size() == 1 )
      {
        vital::track_sptr track = std::get<2>( result )[0];
        unique_track_id_t track_id;
        track_id.first = std::get<0>( result );
        track_id.second = track->id();

        auto it = top_results.find( track_id );
        if( it != top_results.end() )
        {
          if( d->use_tracks_for_history)
          {
            if( it->second->relevancy_score() >= result_score )
            {
              continue;
            }
            else
            {
              entry = it->second;
              insert = false;
            }
          }
          else
          {
            entry = it->second;
            insert = false;
            vital::track_descriptor_sptr entry_descriptor = (*entry->descriptors())[0];
            auto hist = std::get<1>( result )->get_history();
            auto entry_hist = entry_descriptor->get_history();

            merge_history( entry_hist, hist );

            entry_descriptor->set_history( entry_hist );
          }
        }
        else
        {
          entry.reset( new vital::query_result() );
          top_results[ track_id ] = entry;
        }
      }

      if( ! entry )
      {
        entry.reset( new vital::query_result() );
      }

      entry->set_query_id( d->active_uid );
      entry->set_stream_id( std::get<0>( result ) );
      entry->set_instance_id( iid );
      entry->set_relevancy_score( result_score );

      // Assign track descriptor set to result
      vital::track_descriptor_set_sptr desc_set = entry->descriptors();
      if( ! desc_set )
      {
        desc_set.reset( new vital::track_descriptor_set() );

        desc_set->push_back( std::get<1>( result ) );
        entry->set_descriptors( desc_set );
      }

      // Assign temporal bounds to this query result
      vital::timestamp ts1, ts2;
      bool is_first = true;

      for( auto desc : *desc_set )
      {
        for( auto hist : desc->get_history() )
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
        new vital::object_track_set( std::get<2>( result ) ) );

      entry->set_tracks( trk_set );

      // Remember this descriptor result for future iterations
      d->previous_results[ entry->instance_id() ] = entry;

      if( insert )
      {
        output->push_back( entry );
      }
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
  push_to_port_using_trait( iqr_model, model );
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
  declare_input_port_using_trait( iqr_model, optional );

  // -- output --
  declare_output_port_using_trait( query_result, optional );
  declare_output_port_using_trait( iqr_model, optional );
}


// -----------------------------------------------------------------------------
void perform_query_process
::make_config()
{
  declare_config_using_trait( external_handler );
  declare_config_using_trait( external_pipeline_file );
  declare_config_using_trait( database_folder );
  declare_config_using_trait( max_result_count );
  declare_config_using_trait( descriptor_postfix );
  declare_config_using_trait( track_postfix );
  declare_config_using_trait( index_postfix );
  declare_config_using_trait( use_tracks_for_history );
  declare_config_using_trait( merge_duplicate_results );
}


// =============================================================================
perform_query_process::priv
::priv( perform_query_process* p )
 : parent( p )
 , external_handler( true )
 , external_pipeline_file( "" )
 , database_folder( "" )
 , use_tracks_for_history( false )
 , merge_duplicate_results( false )
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
