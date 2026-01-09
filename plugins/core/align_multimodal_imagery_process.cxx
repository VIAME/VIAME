/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Register multi-modal images.
 */

#include "align_multimodal_imagery_process.h"

#include <vital/vital_types.h>

#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/homography.h>

#include <sstream>
#include <iostream>
#include <list>
#include <limits>
#include <cmath>


namespace viame
{

namespace core
{

create_config_trait( output_frames_without_match, bool, "true",
  "Output frames without any valid matches" );
create_config_trait( max_time_offset, double, "0.5",
  "The maximum time difference (s) under whitch two frames can be tested" );

//------------------------------------------------------------------------------
// Private implementation class
class align_multimodal_imagery_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  bool m_output_frames_without_match;
  double m_max_time_offset;

  // Internal buffer
  std::list< buffered_frame > m_optical_frames;
  std::list< buffered_frame > m_thermal_frames;

  kwiver::vital::timestamp m_last_optical_ts;
  kwiver::vital::timestamp m_last_thermal_ts;

  bool m_optical_finished;
  bool m_thermal_finished;

  bool m_check_optical;
  bool m_check_thermal;
};

// =============================================================================

align_multimodal_imagery_process
::align_multimodal_imagery_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new align_multimodal_imagery_process::priv() )
{
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}


align_multimodal_imagery_process
::~align_multimodal_imagery_process()
{
}


// -----------------------------------------------------------------------------
void
align_multimodal_imagery_process
::_configure()
{
  d->m_output_frames_without_match =
    config_value_using_trait( output_frames_without_match );
  d->m_max_time_offset =
    config_value_using_trait( max_time_offset ) * 1e6;
}


// -----------------------------------------------------------------------------
void
align_multimodal_imagery_process
::_step()
{
  kwiver::vital::timestamp optical_time;
  kwiver::vital::image_container_sptr optical_image;
  std::string optical_file_name;

  kwiver::vital::timestamp thermal_time;
  kwiver::vital::image_container_sptr thermal_image;
  std::string thermal_file_name;

  // Check for completion of optical frame stream
  if( !d->m_optical_finished )
  {
    auto port_info = peek_at_port_using_trait( optical_image );

    if( port_info.datum->type() == sprokit::datum::complete )
    {
      d->m_optical_finished = true;
      grab_edge_datum_using_trait( optical_image );

      if( has_input_port_edge_using_trait( optical_timestamp ) )
      {
        grab_edge_datum_using_trait( optical_timestamp );
      }
      if( has_input_port_edge_using_trait( optical_file_name ) )
      {
        grab_edge_datum_using_trait( optical_file_name );
      }
    }
  }

  // Read optical frame
  if( d->m_check_optical )
  {
    if( !d->m_optical_finished &&
        has_input_port_edge_using_trait( optical_image ) )
    {
      optical_image = grab_from_port_using_trait( optical_image );
    }

    if( !d->m_optical_finished &&
        has_input_port_edge_using_trait( optical_timestamp ) )
    {
      optical_time = grab_from_port_using_trait( optical_timestamp );
  
      LOG_DEBUG( logger(), "Received optical frame " << optical_time );
    }

    if( !d->m_optical_finished &&
        has_input_port_edge_using_trait( optical_file_name ) )
    {
      optical_file_name = grab_from_port_using_trait( optical_file_name );
    }
  }

  // Check for completion of thermal frame stream
  if( !d->m_thermal_finished )
  {
    auto port_info = peek_at_port_using_trait( thermal_image );

    if( port_info.datum->type() == sprokit::datum::complete )
    {
      d->m_thermal_finished = true;
      grab_edge_datum_using_trait( thermal_image );

      if( has_input_port_edge_using_trait( thermal_timestamp ) )
      {
        grab_edge_datum_using_trait( thermal_timestamp );
      }
      if( has_input_port_edge_using_trait( thermal_file_name ) )
      {
        grab_edge_datum_using_trait( thermal_file_name );
      }
    }
  }

  // Read thermal frame
  if( d->m_check_thermal )
  {
    if( !d->m_thermal_finished &&
        has_input_port_edge_using_trait( thermal_image ) )
    {
      thermal_image = grab_from_port_using_trait( thermal_image );
    }

    if( !d->m_thermal_finished &&
        has_input_port_edge_using_trait( thermal_timestamp ) )
    {
      thermal_time = grab_from_port_using_trait( thermal_timestamp );
  
      LOG_DEBUG( logger(), "Received thermal frame " << thermal_time );
    }

    if( !d->m_thermal_finished &&
        has_input_port_edge_using_trait( thermal_file_name ) )
    {
      thermal_file_name = grab_from_port_using_trait( thermal_file_name );
    }
  }

  // Determine dominant type
  bool optical_dominant = true;

  if( count_output_port_edges_using_trait( warped_optical_image ) > 0 )
  {
    optical_dominant = false;

    if( count_output_port_edges_using_trait( warped_thermal_image ) > 0 )
    {
      throw std::runtime_error( "Cannot connect both warp image ports" );
    }
  }

  // Check for timestamp consistency
  if( optical_time.has_valid_time() && thermal_time.has_valid_time() )
  {
    if( d->m_last_optical_ts.has_valid_time() &&
        d->m_last_thermal_ts.has_valid_time() &&
        ( d->m_last_optical_ts > optical_time ||
          d->m_last_thermal_ts > thermal_time ) )
    {
      if( optical_time.get_time_usec() == thermal_time.get_time_usec() )
      {
        d->m_optical_frames.clear();
        d->m_thermal_frames.clear();
      }
      else
      {
        throw std::runtime_error( "Frames must be inserted in chronological order" );
      }
    }

    d->m_last_optical_ts = optical_time;
    d->m_last_thermal_ts = thermal_time;
  }

  // Add images to buffer
  if( optical_image )
  {
    d->m_optical_frames.push_back(
      buffered_frame( optical_image, optical_time, optical_file_name ) );
  }

  if( thermal_image )
  {
    d->m_thermal_frames.push_back(
      buffered_frame( thermal_image, thermal_time, thermal_file_name ) );
  }

  // Determine if any images need to be tested
  std::list< buffered_frame >* dom =
    ( optical_dominant ? &d->m_optical_frames : &d->m_thermal_frames );
  std::list< buffered_frame >* sub =
    ( optical_dominant ? &d->m_thermal_frames : &d->m_optical_frames );

  const bool this_is_the_end = ( d->m_optical_finished && d->m_thermal_finished );

  d->m_check_optical = true;
  d->m_check_thermal = true;

  if( !dom->empty() )
  {
    for( auto dom_entry = dom->begin(); dom_entry != dom->end(); )
    {
      bool found_in_range = false; // Crit1: A sub-frame in search interval found
      bool minimum_found = false;  // Crit2: Found guaranteed local minimum diff between times
      bool bound_exceeded = false; // Crit3: Upper search bound exceeded
      bool is_exact = false;       // Crit4: Times match exactly

      const double lower_time = dom_entry->time() - d->m_max_time_offset;
      const double upper_time = dom_entry->time() + d->m_max_time_offset;

      double min_dist = std::numeric_limits< double >::max();
      buffered_frame* closest_frame = NULL;

      // Special case to prevent over-buffering
      if( dom_entry == dom->begin() &&
          sub->size() == 1 && sub->begin()->time() < lower_time )
      {
        sub->erase( sub->begin() );
        d->m_check_optical = !optical_dominant;
        d->m_check_thermal = optical_dominant;
        break;
      }

      // Iterate over sub
      for( auto sub_entry = sub->begin(); sub_entry != sub->end(); )
      {
        if( sub_entry->time() < lower_time )
        {
          sub_entry = sub->erase( sub_entry );
        }
        else if( sub_entry->time() > upper_time )
        {
          minimum_found = found_in_range;
          bound_exceeded = true;
          break;
        }
        else
        {
          // This frame is in the desired time range
          found_in_range = true;

          double abs_diff = std::fabs( dom_entry->time() - sub_entry->time() );

          if( abs_diff < min_dist )
          {
            min_dist = abs_diff;
            closest_frame = &(*sub_entry);

            if( abs_diff == 0 )
            {
              is_exact = true;
              break;
            }
          }
          else
          {
            minimum_found = true;
            break;
          }

          sub_entry++;
        }
      }

      // Definite match 
      if( found_in_range && ( minimum_found || this_is_the_end || is_exact ) )
      {
        if( optical_dominant )
        {
          attempt_registration( *dom_entry, *closest_frame, true );
        }
        else
        {
          attempt_registration( *closest_frame, *dom_entry, false );
        }
        dom_entry = dom->erase( dom_entry );
      }
      // No match for this frame ever
      else if( ( bound_exceeded && !found_in_range ) || this_is_the_end )
      {
        if( optical_dominant )
        {
          output_no_match( *dom_entry, 0 );
        }
        else
        {
          output_no_match( *dom_entry, 1 );
        }
        dom_entry = dom->erase( dom_entry );
      }
      // No match found yet...  wait until we receive more frames or EOV
      else
      {
        break;
      }
    }
  }

  if( this_is_the_end )
  {
    // Send complete messages, shut down
    mark_process_as_complete();

    const sprokit::datum_t dat = sprokit::datum::complete_datum();

    push_datum_to_port_using_trait( optical_image, dat );
    push_datum_to_port_using_trait( optical_file_name, dat );
    push_datum_to_port_using_trait( thermal_image, dat );
    push_datum_to_port_using_trait( thermal_file_name, dat );
    push_datum_to_port_using_trait( timestamp, dat );
    push_datum_to_port_using_trait( warped_optical_image, dat );
    push_datum_to_port_using_trait( warped_thermal_image, dat );
    push_datum_to_port_using_trait( optical_to_thermal_homog, dat );
    push_datum_to_port_using_trait( thermal_to_optical_homog, dat );
    push_datum_to_port_using_trait( success_flag, dat );
  }
}


// -----------------------------------------------------------------------------
void
align_multimodal_imagery_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( optical_image, required );
  declare_input_port_using_trait( optical_timestamp, required );
  declare_input_port_using_trait( optical_file_name, optional );
  declare_input_port_using_trait( thermal_image, required );
  declare_input_port_using_trait( thermal_timestamp, required );
  declare_input_port_using_trait( thermal_file_name, optional );

  // -- output --
  declare_output_port_using_trait( optical_image, optional );
  declare_output_port_using_trait( optical_file_name, optional );
  declare_output_port_using_trait( thermal_image, optional );
  declare_output_port_using_trait( thermal_file_name, optional );
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( warped_optical_image, optional );
  declare_output_port_using_trait( warped_thermal_image, optional );
  declare_output_port_using_trait( optical_to_thermal_homog, optional );
  declare_output_port_using_trait( thermal_to_optical_homog, optional );
  declare_output_port_using_trait( success_flag, optional );
}


// -----------------------------------------------------------------------------
void
align_multimodal_imagery_process
::make_config()
{
  declare_config_using_trait( output_frames_without_match );
  declare_config_using_trait( max_time_offset );
}


// =============================================================================
align_multimodal_imagery_process::priv
::priv()
  : m_optical_finished( false )
  , m_thermal_finished( false )
  , m_check_optical( true )
  , m_check_thermal( true )
{
}


align_multimodal_imagery_process::priv
::~priv()
{
}


void
align_multimodal_imagery_process
::attempt_registration( const buffered_frame& frame1,
                        const buffered_frame& frame2,
                        const bool output_frame1_time )
{
  // Output required elements depending on connections
  this->push_to_port_using_trait( optical_image,
    frame1.image );
  this->push_to_port_using_trait( optical_file_name,
    frame1.name );
  this->push_to_port_using_trait( thermal_image,
    frame2.image );
  this->push_to_port_using_trait( thermal_file_name,
    frame2.name );
  this->push_to_port_using_trait( timestamp,
     output_frame1_time ? frame1.ts : frame2.ts );
  this->push_to_port_using_trait( warped_optical_image,
    kwiver::vital::image_container_sptr() );
  this->push_to_port_using_trait( warped_thermal_image,
    kwiver::vital::image_container_sptr() );
  this->push_to_port_using_trait( optical_to_thermal_homog,
    kwiver::vital::homography_sptr() );
  this->push_to_port_using_trait( thermal_to_optical_homog,
    kwiver::vital::homography_sptr() );
  this->push_to_port_using_trait( success_flag,
    true );
}


void
align_multimodal_imagery_process
::output_no_match( const buffered_frame& frame, unsigned stream_id )
{
  if( !d->m_output_frames_without_match )
  {
    return;
  }

  if( stream_id == 0 )
  {
    this->push_to_port_using_trait( optical_image,
      frame.image );
    this->push_to_port_using_trait( thermal_image,
      kwiver::vital::image_container_sptr() );
    this->push_to_port_using_trait( optical_file_name,
      frame.name );
    this->push_to_port_using_trait( thermal_file_name,
      std::string() );
  }
  else if( stream_id == 1 )
  {
    this->push_to_port_using_trait( optical_image,
      kwiver::vital::image_container_sptr() );
    this->push_to_port_using_trait( thermal_image,
      frame.image );
    this->push_to_port_using_trait( optical_file_name,
      std::string() );
    this->push_to_port_using_trait( thermal_file_name,
      frame.name );
  }
  else
  {
    throw std::runtime_error( "Invalid index" );
  }

  this->push_to_port_using_trait( timestamp,
    frame.ts );
  this->push_to_port_using_trait( warped_optical_image,
    kwiver::vital::image_container_sptr() );
  this->push_to_port_using_trait( warped_thermal_image,
    kwiver::vital::image_container_sptr() );
  this->push_to_port_using_trait( optical_to_thermal_homog,
    kwiver::vital::homography_sptr() );
  this->push_to_port_using_trait( thermal_to_optical_homog,
    kwiver::vital::homography_sptr() );
  this->push_to_port_using_trait( success_flag,
    false );
}


} // end namespace core

} // end namespace viame
