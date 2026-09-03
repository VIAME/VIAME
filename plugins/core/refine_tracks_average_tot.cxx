/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "refine_tracks_average_tot.h"

#include "utilities_target_clfr.h"

#include <stdexcept>
#include <vector>

namespace viame {

namespace kv = kwiver::vital;

namespace {

const std::string PASSTHROUGH_OPTION = "detection";

bool
is_valid_tot_option( const std::string& option )
{
  return option == PASSTHROUGH_OPTION ||
         option == "average" ||
         option == "weighted_average" ||
         option == "weighted_average_scaled_by_conf";
}

kv::detected_object_type_sptr
average_track_type( const kv::track_sptr& trk,
                    bool weighted,
                    bool scale_by_conf,
                    const std::string& ignore_class )
{
  std::vector< kv::detected_object_sptr > detections;

  for( auto state : *trk )
  {
    auto ots = kv::object_track_state::downcast( state );

    if( ots && ots->detection() )
    {
      detections.push_back( ots->detection() );
    }
  }

  return core::compute_average_classification(
    detections, weighted, scale_by_conf, ignore_class );
}

} // end anonymous namespace

// -----------------------------------------------------------------------------
void
refine_tracks_average_tot
::set_configuration_internal( kv::config_block_sptr config )
{
  if( !is_valid_tot_option( c_tot_option ) )
  {
    throw std::runtime_error(
      "Invalid tot_option \"" + c_tot_option + "\", must be one of: "
      "detection, average, weighted_average, "
      "weighted_average_scaled_by_conf" );
  }
}

// -----------------------------------------------------------------------------
bool
refine_tracks_average_tot
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// -----------------------------------------------------------------------------
kv::object_track_set_sptr
refine_tracks_average_tot
::refine( kv::timestamp ts,
          kv::image_container_sptr image_data,
          kv::object_track_set_sptr tracks ) const
{
  if( tracks && c_tot_option != PASSTHROUGH_OPTION )
  {
    for( auto trk : tracks->tracks() )
    {
      if( trk && !trk->empty() )
      {
        m_tracks[ trk->id() ] = trk;
      }
    }
  }

  return tracks;
}

// -----------------------------------------------------------------------------
kv::object_track_set_sptr
refine_tracks_average_tot
::finalize() const
{
  if( c_tot_option == PASSTHROUGH_OPTION || m_tracks.empty() )
  {
    return kv::object_track_set_sptr();
  }

  const bool weighted =
    c_tot_option.find( "weighted" ) != std::string::npos;
  const bool scale_by_conf =
    c_tot_option.find( "scaled_by_conf" ) != std::string::npos;

  std::vector< kv::track_sptr > output;
  output.reserve( m_tracks.size() );

  for( auto trk_pair : m_tracks )
  {
    auto averaged = average_track_type(
      trk_pair.second, weighted, scale_by_conf, c_tot_ignore_class );

    if( !averaged )
    {
      output.push_back( trk_pair.second );
      continue;
    }

    // Cloned so the averaged type never feeds back into whatever upstream
    // stage still holds these detections.
    auto trk = trk_pair.second->clone( kv::clone_type::DEEP );

    for( auto state : *trk )
    {
      auto ots = kv::object_track_state::downcast( state );

      if( ots && ots->detection() )
      {
        ots->detection()->set_type( averaged );
      }
    }

    output.push_back( trk );
  }

  m_tracks.clear();

  return std::make_shared< kv::object_track_set >( output );
}

} // end namespace
