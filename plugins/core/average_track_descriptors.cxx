/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "average_track_descriptors.h"

#include <deque>
#include <utility>

namespace viame {

namespace kv = kwiver::vital;

// ==================================================================================
class average_track_descriptors::priv
{
public:
  priv( average_track_descriptors* parent )
    : m_parent( parent )
    , m_logger( kv::get_logger( "average_track_descriptors" ) )
    , m_rolling( false )
    , m_interval( 5 )
  { }

  ~priv() { }

  average_track_descriptors* m_parent;
  kv::logger_handle_t m_logger;
  bool m_rolling;
  unsigned int m_interval;

  std::map< kv::track_id_t, std::deque< std::vector< double > > > m_history;
};


// ==================================================================================
average_track_descriptors
::average_track_descriptors()
  : d( new priv( this ) )
{
}


// ----------------------------------------------------------------------------------
average_track_descriptors
::~average_track_descriptors()
{
}


// ----------------------------------------------------------------------------------
kv::config_block_sptr
average_track_descriptors
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "rolling", d->m_rolling,
    "When set, produce an output for each input as the rolling average"
    " of the last N descriptors, where N is the interval.  When reset,"
    " produce an output only for the first input and then every Nth input"
    " thereafter for any given track." );
  config->set_value( "interval", d->m_interval,
    "When the interval is N, every descriptor output (after the first N inputs)"
    " is based on the last N descriptors seen as input for the given track." );

  return config;
}


// ----------------------------------------------------------------------------------
void
average_track_descriptors
::set_configuration( kv::config_block_sptr config )
{
  d->m_rolling = config->get_value< bool >( "rolling", d->m_rolling );
  d->m_interval = config->get_value< unsigned int >( "interval", d->m_interval );
}


// ----------------------------------------------------------------------------------
bool
average_track_descriptors
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// ----------------------------------------------------------------------------------
kv::track_descriptor_set_sptr
average_track_descriptors
::compute( kv::timestamp ts,
           kv::image_container_sptr image_data,
           kv::object_track_set_sptr tracks )
{
  kv::track_descriptor_set_sptr tds( new kv::track_descriptor_set() );

  if( tracks )
  {
    for( kv::track_sptr track : tracks->tracks() )
    {
      if( !track )
      {
        LOG_WARN( d->m_logger, "Warning: Invalid Track" );
        continue;
      }

      kv::track::history_const_itr it = track->find( ts.get_frame() );
      if( it != track->end() )
      {
        std::shared_ptr< kv::object_track_state > ots =
          std::dynamic_pointer_cast< kv::object_track_state >( *it );

        if( ots && ots->detection() && ots->detection()->descriptor() )
        {
          std::vector< double > descriptor = ots->detection()->descriptor()->as_double();
          std::vector< double > average;
          if( !d->m_rolling && !d->m_history.count( track->id() ) )
          {
            // This is the first detection in the track, so output it
            // as-is.  This ensures that every input track has a
            // corresponding output track.
            average = std::move( descriptor );
            // Create an empty deque in the map
            d->m_history[ track->id() ];
          }
          else
          {
            std::deque< std::vector< double > >& average_history = d->m_history[ track->id() ];

            average_history.push_back( std::move( descriptor ) );
            if( !d->m_rolling && average_history.size() != d->m_interval )
            {
              continue;
            }
            else
            {
              std::size_t i = 0;

              bool done = false;
              while( true )
              {
                double total = 0.0;

                for( std::vector< double >& entry : average_history )
                {
                  if( i >= entry.size() )
                  {
                    done = true;
                    break;
                  }

                  total += entry[ i ];
                }

                if( done )
                {
                  break;
                }
                else
                {
                  average.push_back( total / average_history.size() );
                  i++;
                }
              }

              if( d->m_rolling )
              {
                if( average_history.size() == d->m_interval )
                {
                  average_history.pop_front();
                }
              }
              else
              {
                average_history.clear();
              }
            }
          }

          kv::track_descriptor_sptr td = kv::track_descriptor::create( "cnn_descriptor" );

          td->add_track_id( track->id() );

          kv::track_descriptor::descriptor_data_sptr data(
            new kv::track_descriptor::descriptor_data_t( average.size() ) );
          std::copy( average.begin(), average.end(), data->raw_data() );
          td->set_descriptor( data );

          // Make history entry
          kv::track_descriptor::history_entry he( ts, ots->detection()->bounding_box() );
          td->add_history_entry( he );

          tds->push_back( td );
        }
      }
    }
  }

  return tds;
}


// ----------------------------------------------------------------------------------
kv::track_descriptor_set_sptr
average_track_descriptors
::flush()
{
  return kv::track_descriptor_set_sptr( new kv::track_descriptor_set() );
}


} // end namespace viame
