/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "average_track_descriptors.h"

#include <utility>

namespace viame {

namespace kv = kwiver::vital;


// ----------------------------------------------------------------------------------
void
average_track_descriptors
::initialize()
{
  m_logger = kv::get_logger( "average_track_descriptors" );
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
        LOG_WARN( m_logger, "Warning: Invalid Track" );
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
          if( !c_rolling && !m_history.count( track->id() ) )
          {
            // This is the first detection in the track, so output it
            // as-is.  This ensures that every input track has a
            // corresponding output track.
            average = std::move( descriptor );
            // Create an empty deque in the map
            m_history[ track->id() ];
          }
          else
          {
            std::deque< std::vector< double > >& average_history = m_history[ track->id() ];

            average_history.push_back( std::move( descriptor ) );
            if( !c_rolling && average_history.size() != c_interval )
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

              if( c_rolling )
              {
                if( average_history.size() == c_interval )
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
