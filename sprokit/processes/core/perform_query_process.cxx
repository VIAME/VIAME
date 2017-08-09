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

    entry->set_stream_id( "/data/virat/video/aphill/09172008flight1tape1_5.mpg" );
    entry->set_instance_id( i );
    entry->set_relevancy_score( ( 4 - i ) * 0.30 );

    typedef vital::track_descriptor td;

    vital::track_descriptor_set_sptr output( new vital::track_descriptor_set() );
    vital::track_descriptor_sptr new_desc = td::create( "cnn_descriptor" );

    td::descriptor_data_sptr_t data( new td::descriptor_data_t( 100 ) );

    for( unsigned i = 0; i < 100; i++ )
    {
      (data->raw_data())[i] = static_cast<double>( i );
    }

    new_desc->set_descriptor( data );

    td::history_entry::image_bbox_t region( 0, 0, 400, 400 );
    td::history_entry hist_entry( vital::timestamp( 0, 0 ), region );
    new_desc->add_history_entry( hist_entry );

    if( i == 1 )
    {

    }
    else if( i == 2 )
    {
      
    }
    else if( i == 3 )
    {
      
    }

    //output->push_back( entry );
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
