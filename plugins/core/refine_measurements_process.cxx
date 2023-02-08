/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
 * \brief Consolidate the output of multiple object trackers
 */

#include "refine_measurements_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>

#include <sprokit/processes/kwiver_type_traits.h>


namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( recompute_all, bool, "false",
  "If set, recompute lengths for all detections using GSD, even those "
  "already containing lengths." );
create_config_trait( output_multiple, bool, "false",
  "Allow outputting multiple possible lengths for each detection"  );
create_config_trait( output_conf_level, bool, "false",
  "Output length confidence metric"  );
create_config_trait( min_valid, double, "-1.0",
  "Minimum allowed valid measurement"  );
create_config_trait( max_valid, double, "-1.0",
  "Maximum allowed valid measurement"  );
create_config_trait( history_length, unsigned, "0",
  "History to consider when averaging GSDs" );
create_config_trait( exp_factor, double, "-1.0",
  "Exponential averaging factor to consider when averaging" );
create_config_trait( border_factor, unsigned, "0",
  "Treat detections this many pixels near image border as ambiguous" );
create_config_trait( percentile, double, "0.45",
  "Percentile GSD to use when combining multiple estimates" );

// =============================================================================
// Private implementation class
class refine_measurements_process::priv
{
public:
  explicit priv( refine_measurements_process* parent );
  ~priv();

  // Configuration settings
  bool m_recompute_all;
  bool m_output_multiple;
  bool m_output_conf_level;
  double m_min_valid;
  double m_max_valid;
  unsigned m_history_length;
  double m_exp_factor;
  unsigned m_border_factor;
  double m_percentile;

  // Internal variables
  double m_last_gsd;
  std::vector< double > m_history;

  // Other variables
  refine_measurements_process* parent;

  // Helper functions
  double percentile( std::vector< double >& vec  );
  bool is_border( const kv::bounding_box_d& box, unsigned w, unsigned h );
};


// -----------------------------------------------------------------------------
refine_measurements_process::priv
::priv( refine_measurements_process* ptr )
  : m_recompute_all( false )
  , m_output_multiple( false )
  , m_output_conf_level( false )
  , m_min_valid( -1.0 )
  , m_max_valid( -1.0 )
  , m_history_length( 0 )
  , m_exp_factor( -1.0 )
  , m_border_factor( 0 )
  , m_percentile( 0.45 )
  , m_last_gsd( -1.0 )
  , m_history()
  , parent( ptr )
{
}


refine_measurements_process::priv
::~priv()
{
}


double
refine_measurements_process::priv
::percentile( std::vector< double >& vec )
{
  if( vec.empty() || m_percentile < 1.0 )
  {
    return -1.0;
  }

  std::sort( vec.begin(), vec.end() );
  unsigned ind = static_cast< unsigned >( m_percentile * vec.size() );
  return vec[ ind ];
}

bool
refine_measurements_process::priv
::is_border( const kv::bounding_box_d& box, unsigned w, unsigned h )
{
  return ( box.min_x() <= m_border_factor ||
           box.min_y() <= m_border_factor ||
           box.max_x() >= w - m_border_factor ||
           box.max_y() >= h - m_border_factor );
}


// =============================================================================
refine_measurements_process
::refine_measurements_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new refine_measurements_process::priv( this ) )
{
  make_ports();
  make_config();

  set_data_checking_level( check_valid );
}


refine_measurements_process
::~refine_measurements_process()
{
}


// -----------------------------------------------------------------------------
void
refine_measurements_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( detected_object_set, optional );
  declare_input_port_using_trait( object_track_set, optional );

  // -- outputs --
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( detected_object_set, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::make_config()
{
  declare_config_using_trait( recompute_all );
  declare_config_using_trait( output_multiple );
  declare_config_using_trait( output_conf_level );
  declare_config_using_trait( min_valid );
  declare_config_using_trait( max_valid );
  declare_config_using_trait( history_length );
  declare_config_using_trait( exp_factor );
  declare_config_using_trait( border_factor );
  declare_config_using_trait( percentile );
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::_configure()
{
  d->m_recompute_all = config_value_using_trait( recompute_all );
  d->m_output_multiple = config_value_using_trait( output_multiple );
  d->m_output_conf_level = config_value_using_trait( output_conf_level );
  d->m_min_valid = config_value_using_trait( min_valid );
  d->m_max_valid = config_value_using_trait( max_valid );
  d->m_history_length = config_value_using_trait( history_length );
  d->m_exp_factor = config_value_using_trait( exp_factor );
  d->m_border_factor = config_value_using_trait( border_factor );
  d->m_percentile = config_value_using_trait( percentile );
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::_step()
{
  kv::object_track_set_sptr input_tracks;
  kv::detected_object_set_sptr input_dets;
  kv::image_container_sptr image;
  kv::timestamp timestamp;

  auto port_info = peek_at_port_using_trait( detected_object_set );

  if( port_info.datum->type() == sprokit::datum::complete )
  {
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( detected_object_set, dat );
    push_datum_to_port_using_trait( object_track_set, dat );
    return;
  }

  if( has_input_port_edge_using_trait( detected_object_set ) )
  {
    input_dets = grab_from_port_using_trait( detected_object_set );
  }
  if( has_input_port_edge_using_trait( object_track_set ) )
  {
    input_tracks = grab_from_port_using_trait( object_track_set );
  }
  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );
  }
  if( has_input_port_edge_using_trait( image ) )
  {
    image = grab_from_port_using_trait( image );
  }

  const unsigned detection_count = ( input_dets ? input_dets->size() : 0 );

  const unsigned img_height = ( image ? image->height() : 0 );
  const unsigned img_width = ( image ? image->width() : 0 );

  std::vector< unsigned > length_conf( detection_count, 0 );
  std::vector< double > gsd_estimates( detection_count, -1.0 );

  const std::string conf_str[5] = { "none", "very_low", "low", "medium", "high" };
  std::vector< double > conf_ests[5];
  unsigned highest_conf = 0;

  if( input_dets )
  {
    unsigned ind = 0;

    for( auto det : *input_dets )
    {
      if( !det->notes().empty() && det->bounding_box().width() > 0 )
      {
        for( auto note : det->notes() )
        {
          if( note.size() > 8 && note.substr( 0, 8 ) == ":length=" )
          {
            double lth = std::stod( note.substr( 8 ) );
            double est = lth / det->bounding_box().width();

            gsd_estimates[ ind ] = est;

            if( ( d->m_min_valid > 0.0 && lth < d->m_min_valid ) ||
                ( d->m_max_valid > 0.0 && lth > d->m_max_valid ) )
            {
              length_conf[ ind ] = 1;
              highest_conf = std::max( highest_conf, 1u );
              conf_ests[1].push_back( est );
            }
            else if( d->is_border( det->bounding_box(), img_width, img_height ) )
            {
              length_conf[ ind ] = 2;
              highest_conf = std::max( highest_conf, 2u );
              conf_ests[2].push_back( est );
            }
            else
            {
              length_conf[ ind ] = 3;
              highest_conf = std::max( highest_conf, 3u );
              conf_ests[3].push_back( est );
            }
          }
        }
      }

      ind++;
    }
  }

  double initial_gsd_est = -1.0;

  if( highest_conf > 1 )
  {
    initial_gsd_est = d->percentile( conf_ests[ highest_conf ] );
    d->m_last_gsd = initial_gsd_est;
  }
  else if( d->m_last_gsd > 0 )
  {
    initial_gsd_est = d->m_last_gsd;
  }

  double gsd_to_use = -1.0;

  if( input_dets && gsd_to_use > 0.0 )
  {
    unsigned ind = 0;

    for( auto det : *input_dets )
    {
      if( ( d->m_recompute_all ||
            det->notes().empty() ||
            length_conf[ ind ] <= 2 ) &&
          det->bounding_box().width() > 0 )
      {
        if( !d->m_output_multiple )
        {
          det->clear_notes();
        }

        double lth = det->bounding_box().width() * gsd_to_use;
        bool bad = false;

        if( ( d->m_min_valid <= 0.0 || lth >= d->m_min_valid ) &&
            ( d->m_max_valid <= 0.0 || lth <= d->m_max_valid ) )
        {
          det->set_length( lth );
        }
        else
        {
          bad = true;
        }

        if( d->m_output_conf_level && !bad )
        {
          det->add_note( ":length_conf=" + conf_str[ length_conf[ ind ] ] );
        }
      }
      else if( d->m_output_conf_level )
      {
        det->add_note( ":length_conf=" + conf_str[ length_conf[ ind ] ] );
      }

      ind++;
    }
  }

  push_to_port_using_trait( detected_object_set, input_dets );
  push_to_port_using_trait( timestamp, timestamp );
}

} // end namespace core

} // end namespace viame
