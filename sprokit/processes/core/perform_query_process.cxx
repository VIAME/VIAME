/*ckwg +29
 * Copyright 2017, 2020 by Kitware, Inc.
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

namespace kwiver {

namespace algo = vital::algo;

create_config_trait( external_handler, bool,
  "true", "Whether or not an external query handler is used" );
create_config_trait( external_pipeline_file, std::string,
  "", "External pipeline definition file location" );
create_config_trait( augmentation_pipeline_file, std::string,
  "", "Augmentation pipeline definition file location" );
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
create_config_trait( unused_descriptors_as_negative, bool,
  "true", "Use un-marked user descriptors as negative exemplars" );
create_config_trait( use_tracks_for_history, bool,
  "false", "Use object track states for track descriptor history" );
create_config_trait( merge_duplicate_results, bool,
  "false", "If use_tracks_for_history is on, use the track with the "
  "highest confidence, otherwise concatenate the history of all entries "
  "with the same track" );

create_algorithm_name_config_trait( descriptor_query );

create_port_trait( feedback_request, query_result, "Feedback requests" );


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
  std::string augmentation_pipeline_file;
  std::string database_folder;

  std::string track_postfix;
  std::string descriptor_postfix;
  std::string index_postfix;
  bool unused_descriptors_as_negative;
  bool use_tracks_for_history;
  bool merge_duplicate_results;

  std::unique_ptr< embedded_pipeline > external_pipeline;
  std::unique_ptr< embedded_pipeline > augmentation_pipeline;

  unsigned max_result_count;

  bool is_first;
  std::map< unsigned, vital::query_result_sptr > previous_results;
  std::map< std::string, unsigned > result_instance_ids;
  std::map< std::string, unsigned > feedback_instance_ids;
  std::map< unsigned, vital::query_result_sptr > forced_positives;
  std::map< unsigned, vital::query_result_sptr > forced_negatives;
  vital::uid active_uid;

  unsigned result_counter;
  bool database_populated;

  algo::query_track_descriptor_set_sptr descriptor_query;

  vital::image_container_set_sptr query_images;
  vital::track_descriptor_set_sptr all_descriptors;

  void reset_query( const vital::database_query_sptr& query );

  unsigned get_instance_id( std::map< std::string, unsigned >& instance_ids,
                            const std::string& uid );

  void add_results_to_list( const vital::query_result_set_sptr& results,
                            const std::vector<std::string>& uids,
                            const std::vector<double>& scores,
                            std::map< std::string, unsigned >& instance_ids,
                            bool feedback_request );
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

  if( d->augmentation_pipeline )
  {
    d->augmentation_pipeline->send_end_of_input();
    d->augmentation_pipeline->receive();
    d->augmentation_pipeline->wait();
    d->augmentation_pipeline.reset();
  }
}


// -----------------------------------------------------------------------------
void perform_query_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  d->external_handler = config_value_using_trait( external_handler );
  d->external_pipeline_file = config_value_using_trait( external_pipeline_file );
  d->augmentation_pipeline_file = config_value_using_trait( augmentation_pipeline_file );
  d->database_folder = config_value_using_trait( database_folder );
  d->max_result_count = config_value_using_trait( max_result_count );
  d->track_postfix = config_value_using_trait( track_postfix );
  d->descriptor_postfix = config_value_using_trait( descriptor_postfix );
  d->index_postfix = config_value_using_trait( index_postfix );
  d->unused_descriptors_as_negative = config_value_using_trait( unused_descriptors_as_negative );
  d->use_tracks_for_history = config_value_using_trait( use_tracks_for_history );
  d->merge_duplicate_results = config_value_using_trait( merge_duplicate_results );

  if( d->external_handler )
  {
    algo::query_track_descriptor_set::set_nested_algo_configuration(
      "descriptor_query",
      algo_config,
      d->descriptor_query );

    if( !d->descriptor_query )
    {
      VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                   "Configuration check failed." );
    }

    algo::query_track_descriptor_set::get_nested_algo_configuration(
      "descriptor_query",
      algo_config,
      d->descriptor_query );

    if( !algo::query_track_descriptor_set::check_nested_algo_configuration(
          "descriptor_query",
          algo_config ) )
    {
      VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                   "Unable to create descriptor query." );
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
      VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                   "Unable to open pipeline file: " + d->external_pipeline_file );
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

  if( !d->augmentation_pipeline_file.empty() )
  {
    std::unique_ptr< embedded_pipeline > new_pipeline =
      std::unique_ptr< embedded_pipeline >( new embedded_pipeline() );

    std::ifstream pipe_stream;
    pipe_stream.open( d->augmentation_pipeline_file, std::ifstream::in );

    if( !pipe_stream )
    {
      VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                   "Unable to open pipeline file: " + d->augmentation_pipeline_file );
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

    d->augmentation_pipeline = std::move( new_pipeline );
    pipe_stream.close();
  }
}


// -----------------------------------------------------------------------------
static void
merge_history( vital::track_descriptor::descriptor_history_t& dest,
               vital::track_descriptor::descriptor_history_t const& src )
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
    else if( src_it == src.end() ||
             src_it->get_timestamp().get_frame() > dest_it->get_timestamp().get_frame() )
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


bool
is_overlap( vital::track_descriptor_sptr p1, vital::track_descriptor_sptr p2 )
{
  // If uncertain return true, they might overlap
  if( !p1 || !p2 )
  {
    return true;
  }

  auto h1 = p1->get_history();
  auto h2 = p2->get_history();

  // If uncertain return true, they might overlap
  if( h1.empty() || h2.empty() )
  {
    return true;
  }

  for( auto e1 : h1 )
  {
    for( auto e2 : h2 )
    {
      if( e1.get_timestamp() == e2.get_timestamp() )
      {
        if( vital::intersection( e1.get_image_location(),
                                 e2.get_image_location() ).area() > 0 )
        {
          return true;
        }
      }
    }
  }

  return false;
}


vital::detected_object_set_sptr
desc_to_det( vital::track_descriptor_set_sptr descs )
{
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set>();

  for( auto desc : *descs )
  {
    if( desc->get_history().size() == 1 )
    {
      detected_set->add(
        std::make_shared< kwiver::vital::detected_object >(
          desc->get_history()[0].get_image_location(),
          1.0 ) );
    }
  }

  return detected_set;
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

    if ( has_input_port_edge_using_trait( iqr_feedback ) )
    {
      grab_edge_datum_using_trait( iqr_feedback );
    }
    if ( has_input_port_edge_using_trait( iqr_model ) )
    {
      grab_edge_datum_using_trait( iqr_model );
    }
    if( has_input_port_edge_using_trait( track_descriptor_set ) )
    {
      grab_edge_datum_using_trait( track_descriptor_set );
    }
    grab_edge_datum_using_trait( image_set );
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( query_result, dat );
    return;
  }

  // Retrieve inputs from ports
  vital::database_query_sptr query;
  vital::iqr_feedback_sptr feedback;
  vital::uchar_vector_sptr model;
  vital::track_descriptor_set_sptr query_descs;
  vital::image_container_set_sptr query_images;
	
  query = grab_from_port_using_trait( database_query );

  if( has_input_port_edge_using_trait( iqr_feedback ) )
  {
    feedback = grab_from_port_using_trait( iqr_feedback );
  }
  if( has_input_port_edge_using_trait( iqr_model ) )
  {
    model = grab_from_port_using_trait( iqr_model );
  }

  query_descs = grab_from_port_using_trait( track_descriptor_set );
  query_images = grab_from_port_using_trait( image_set );

  if( query_descs )
  {
    d->all_descriptors = query_descs;
  }

  if( query_images && !query_images->empty() )
  {
    d->query_images = query_images;
  }

  // Declare output
  vital::query_result_set_sptr results( new vital::query_result_set() );
  vital::query_result_set_sptr feedback_requests( new vital::query_result_set() );

  // No query received, do nothing, return no results
  if( !query && !feedback && !model )
  {
    push_to_port_using_trait( query_result, results );
    push_to_port_using_trait( feedback_request, feedback_requests );
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
    vital::string_vector_sptr iqr_positive_uids( new vital::string_vector() );
    vital::string_vector_sptr iqr_negative_uids( new vital::string_vector() );

    if( feedback &&
      ( !feedback->positive_ids().empty() ) )
    {
      for( auto id : feedback->positive_ids() )
      {
        for( auto desc_sptr : *d->previous_results[id]->descriptors() )
        {
          iqr_positive_uids->push_back( desc_sptr->get_uid().value() );
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
          iqr_negative_uids->push_back( desc_sptr->get_uid().value() );
        }

        d->forced_negatives[ id ] = d->previous_results[ id ];

        auto positive_itr = d->forced_positives.find( id );

        if( positive_itr != d->forced_positives.end() )
        {
          d->forced_positives.erase( positive_itr );
        }
      }
    }

    // Format data to simplified format for external
    std::vector< vital::descriptor_sptr > exemplar_raw_pos_descs;
    std::vector< vital::descriptor_sptr > exemplar_raw_neg_descs;

    vital::string_vector_sptr exemplar_pos_uids( new vital::string_vector() );
    vital::string_vector_sptr exemplar_neg_uids( new vital::string_vector() );

    if( query )
    {
      for( auto track_desc : *query->descriptors() )
      {
        exemplar_pos_uids->push_back( track_desc->get_uid().value() );
        exemplar_raw_pos_descs.push_back( track_desc->get_descriptor() );
      }

      if( !feedback && d->augmentation_pipeline )
      {
        if( d->query_images->empty() )
        {
          throw std::runtime_error( "Must supply images for use with augmentation pipeline" );
        }

        // Run seperate augmentation pipeline to get more positives and negatives
        for( auto query_image = d->query_images->begin();
             query_image == d->query_images->end(); query_image++ )
        {
          auto ids = adapter::adapter_data_set::create();

          vital::descriptor_set_sptr pos_descs(
            new vital::simple_descriptor_set( exemplar_raw_pos_descs ) );

          vital::detected_object_set_sptr pos_dets = desc_to_det( query->descriptors() );

          ids->add_value( "image", *query_image );
          ids->add_value( "positive_descriptors", pos_descs );
          ids->add_value( "positive_detections", pos_dets );

          d->augmentation_pipeline->send( ids );

          auto const& ods = d->augmentation_pipeline->receive();
  
          if( ods->is_end_of_data() )
          {
            throw std::runtime_error( "Pipeline terminated unexpectingly" );
          }

          // Grab result from pipeline output data set
          auto const& iter1 = ods->find( "new_positive_descriptors" );
          auto const& iter2 = ods->find( "new_positive_ids" );
          auto const& iter3 = ods->find( "new_negative_descriptors" );
          auto const& iter4 = ods->find( "new_negative_ids" );

          if( iter1 == ods->end() || iter2 == ods->end() ||
              iter3 == ods->end() || iter4 == ods->end() )
          {
            throw std::runtime_error( "Empty pipeline output" );
          }

          vital::descriptor_set_sptr new_positive_descriptors =
            iter1->second->get_datum< vital::descriptor_set_sptr >();
          vital::string_vector_sptr new_positive_ids =
            iter2->second->get_datum< vital::string_vector_sptr >();
          vital::descriptor_set_sptr new_negative_descriptors =
            iter3->second->get_datum< vital::descriptor_set_sptr >();
          vital::string_vector_sptr new_negative_ids =
            iter4->second->get_datum< vital::string_vector_sptr >();

          const auto& pos_iter = new_positive_descriptors->descriptors();
          const auto& neg_iter = new_positive_descriptors->descriptors();

          if( !pos_iter.empty() )
          {
            exemplar_raw_pos_descs.insert( exemplar_raw_pos_descs.end(),
              pos_iter.begin(), pos_iter.end() );
            exemplar_pos_uids->insert( exemplar_pos_uids->end(), 
              new_positive_ids->begin(), new_positive_ids->end() );
          }

          if( !neg_iter.empty() )
          {
            exemplar_raw_neg_descs.insert( exemplar_raw_neg_descs.end(),
              neg_iter.begin(), neg_iter.end() );
            exemplar_neg_uids->insert( exemplar_neg_uids->end(), 
              new_negative_ids->begin(), new_negative_ids->end() );
          }
        }
      }

      if( !feedback && d->unused_descriptors_as_negative )
      {
        if( !d->all_descriptors )
        {
          throw std::runtime_error( "Must supply descriptors to use with unused as negative option" );
        }

        // Use background descriptors as negative examples
        for( auto desc : *( d->all_descriptors ) )
        {
          bool no_overlap = true;
  
          for( auto comp : *( query->descriptors() ) )
          {
            if( is_overlap( desc, comp ) )
            {
              no_overlap = false;
              break;
            }
          }
  
          if( no_overlap )
          {
            exemplar_neg_uids->push_back( desc->get_uid().value() );
            exemplar_raw_neg_descs.push_back( desc->get_descriptor() );
          }
        }
      }
    }

    vital::descriptor_set_sptr exemplar_pos_descs(
      new vital::simple_descriptor_set( exemplar_raw_pos_descs ) );
    vital::descriptor_set_sptr exemplar_neg_descs(
      new vital::simple_descriptor_set( exemplar_raw_neg_descs ) );

    // Set request on pipeline inputs
    auto ids = adapter::adapter_data_set::create();

    ids->add_value( "positive_descriptor_set", exemplar_pos_descs );
    ids->add_value( "positive_exemplar_uids", exemplar_pos_uids );
    ids->add_value( "negative_descriptor_set", exemplar_neg_descs );
    ids->add_value( "negative_exemplar_uids", exemplar_neg_uids );
    ids->add_value( "iqr_positive_uids", iqr_positive_uids );
    ids->add_value( "iqr_negative_uids", iqr_negative_uids );
    ids->add_value( "iqr_query_model", model );

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
    auto const& iter4 = ods->find( "feedback_uids" );
    auto const& iter5 = ods->find( "feedback_scores" );

    if( iter1 == ods->end() || iter2 == ods->end() || iter3 == ods->end() )
    {
      throw std::runtime_error( "Empty pipeline output" );
    }

    vital::string_vector_sptr result_uids =
      iter1->second->get_datum< vital::string_vector_sptr >();
    vital::double_vector_sptr result_scores =
      iter2->second->get_datum< vital::double_vector_sptr >();

    vital::string_vector_sptr feedback_uids;
    if( iter4 != ods->end() )
    {
      feedback_uids =
        iter4->second->get_datum< vital::string_vector_sptr >();
    }
    else
    {
      feedback_uids.reset( new std::vector<std::string> );
    }

    vital::double_vector_sptr feedback_scores;
    if( iter5 != ods->end() )
    {
      feedback_scores =
        iter5->second->get_datum< vital::double_vector_sptr >();
    }
    else
    {
      feedback_scores.reset( new std::vector<double> );
    }

    model = iter3->second->get_datum< vital::uchar_vector_sptr >();

    // Handle forced positive examples, set score to 1, make sure at front
    for( auto itr = d->forced_positives.begin();
         itr != d->forced_positives.end(); itr++ )
    {
      itr->second->set_relevancy_score( 1.0 );
      results->push_back( itr->second );
    }

    // Handle all new or unadjudacted results
    d->add_results_to_list( results, *result_uids, *result_scores, d->result_instance_ids, false );
    d->add_results_to_list( feedback_requests, *feedback_uids, *feedback_scores, d->feedback_instance_ids, true );

    // Handle forced negative examples, set score to 0, make sure at end of result set
    for( auto itr = d->forced_negatives.begin();
         itr != d->forced_negatives.end(); itr++ )
    {
      itr->second->set_relevancy_score( 0.0 );
      results->push_back( itr->second );
    }
  }
  else
  {
    throw std::runtime_error( "Only external handler mode yet supported" );
  }

  // Push outputs downstream
  push_to_port_using_trait( query_result, results );
  push_to_port_using_trait( feedback_request, feedback_requests );
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
  declare_input_port_using_trait( track_descriptor_set, optional );
  declare_input_port_using_trait( image_set, optional );

  // -- output --
  declare_output_port_using_trait( query_result, optional );
  declare_output_port_using_trait( feedback_request, optional );
  declare_output_port_using_trait( iqr_model, optional );
}


// -----------------------------------------------------------------------------
void perform_query_process
::make_config()
{
  declare_config_using_trait( external_handler );
  declare_config_using_trait( external_pipeline_file );
  declare_config_using_trait( augmentation_pipeline_file );
  declare_config_using_trait( database_folder );
  declare_config_using_trait( max_result_count );
  declare_config_using_trait( descriptor_postfix );
  declare_config_using_trait( track_postfix );
  declare_config_using_trait( index_postfix );
  declare_config_using_trait( unused_descriptors_as_negative );
  declare_config_using_trait( use_tracks_for_history );
  declare_config_using_trait( merge_duplicate_results );
}


// =============================================================================
perform_query_process::priv
::priv( perform_query_process* p )
 : parent( p )
 , external_handler( true )
 , external_pipeline_file( "" )
 , augmentation_pipeline_file( "" )
 , database_folder( "" )
 , unused_descriptors_as_negative( true )
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
::add_results_to_list( const vital::query_result_set_sptr& results,
                       const std::vector<std::string>& uids,
                       const std::vector<double>& scores,
                       std::map< std::string, unsigned >& instance_ids,
                       bool feedback_request )
{
  typedef std::pair< std::string, vital::track_id_t > unique_track_id_t;
  std::map< unique_track_id_t, vital::query_result_sptr > top_results;

  for( unsigned i = 0; i < uids.size(); ++i )
  {
    if( i > max_result_count || (feedback_request && i > 20) )
    {
      break;
    }

    auto uid = uids[i];
    auto score = scores[i];

    vital::algo::query_track_descriptor_set::desc_tuple_t result;
    if( !descriptor_query->get_track_descriptor( uid, result ))
    {
      continue;
    }

    // Create result set and set relevant IDs
    auto iid = get_instance_id( instance_ids, uid );

    // Check if result is forced positive or negative (e.g. annotated by user)
    if( forced_positives.find( iid ) != forced_positives.end() ||
        forced_negatives.find( iid ) != forced_negatives.end() )
    {
      continue;
    }

    vital::query_result_sptr entry;
    bool insert = true;

    // If there is more than one track for a descriptor, there's no point in
    // trying to do any merging
    if( merge_duplicate_results && std::get<2>( result ).size() == 1 )
    {
      vital::track_sptr track = std::get<2>( result )[0];
      unique_track_id_t track_id;
      track_id.first = std::get<0>( result );
      track_id.second = track->id();

      auto it = top_results.find( track_id );
      if( it != top_results.end() )
      {
        if( use_tracks_for_history)
        {
          if( it->second->relevancy_score() >= score )
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

    entry->set_query_id( active_uid );
    entry->set_stream_id( std::get<0>( result ) );
    entry->set_instance_id( iid );
    if( feedback_request )
    {
      entry->set_relevancy_score( 0.0 );
      entry->set_preference_score( score );
    }
    else
    {
      entry->set_relevancy_score( score );
    }

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
    previous_results[ entry->instance_id() ] = entry;

    if( insert )
    {
      results->push_back( entry );
    }
  }
}


void perform_query_process::priv
::reset_query( const vital::database_query_sptr& query )
{
  result_counter = 0;
  result_instance_ids.clear();
  feedback_instance_ids.clear();
  previous_results.clear();
  forced_positives.clear();
  forced_negatives.clear();
  active_uid = query->id();
}


unsigned perform_query_process::priv
::get_instance_id( std::map< std::string, unsigned >& instance_ids, const std::string& uid )
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
