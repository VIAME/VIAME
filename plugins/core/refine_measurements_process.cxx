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

  // Internal variables
  double m_last_gsd;

  // Other variables
  refine_measurements_process* parent;
};


// -----------------------------------------------------------------------------
refine_measurements_process::priv
::priv( refine_measurements_process* ptr )
  : m_recompute_all( false )
  , m_output_multiple( false )
  , m_last_gsd( -1.0 )
  , parent( ptr )
{
}


refine_measurements_process::priv
::~priv()
{
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
}

// -----------------------------------------------------------------------------
void
refine_measurements_process
::_configure()
{
  d->m_recompute_all = config_value_using_trait( recompute_all );
  d->m_output_multiple = config_value_using_trait( output_multiple );
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

  double cumulative_sum = 0.0;
  unsigned sample_count = 0;

  double usable_gsd = -1.0;

  if( input_dets )
  {
    for( auto det : *input_dets )
    {
      if( !det->notes().empty() && det->bounding_box().width() > 0 )
      {
        for( auto note : det->notes() )
        {
          if( note.size() > 8 && note.substr( 0, 8 ) == ":length=" )
          {
            double l = std::stod( note.substr( 8 ) );

            cumulative_sum += ( l / det->bounding_box().width() );
            sample_count++;
          }
        }
      }
    }
  }

  if( sample_count > 0 )
  {
    usable_gsd = cumulative_sum / sample_count;
    d->m_last_gsd = usable_gsd;
  }
  else if( d->m_last_gsd > 0 )
  {
    usable_gsd = d->m_last_gsd;
  }

  if( input_dets && usable_gsd > 0.0 )
  {
    for( auto det : *input_dets )
    {
      if( ( d->m_recompute_all || det->notes().empty() ) &&
          det->bounding_box().width() > 0 )
      {
        if( !d->m_output_multiple )
        {
          det->clear_notes();
        }
        det->set_length( det->bounding_box().width() * usable_gsd );
      }
    }
  }

  push_to_port_using_trait( detected_object_set, input_dets );
  push_to_port_using_trait( timestamp, timestamp );
}

} // end namespace core

} // end namespace viame
