/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * \brief Register images using ITK.
 */

#include "RegistrationProcess.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/homography.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <arrows/ocv/image_container.h>

#include <sstream>
#include <iostream>
#include <list>


namespace viame
{

namespace itk
{

create_config_trait( output_frames_without_match, bool, "true",
  "Output frames without any valid matches" );
create_config_trait( max_time_offset, double, "0.5",
  "The maximum time difference under whitch two frames can be tested" );

create_port_trait( optical_image, image, "Optical image" );
create_port_trait( optical_timestamp, timestamp, "Optical timestamp" );
create_port_trait( optical_file_name, file_name, "Optical file name" );

create_port_trait( thermal_image, image, "Thermal image" );
create_port_trait( thermal_timestamp, timestamp, "Thermal timestamp" );
create_port_trait( thermal_file_name, file_name, "Thermal file name" );

create_port_trait( warped_optical_image, image, "Warped optical image" );
create_port_trait( warped_thermal_image, image, "Warped thermal image" );
create_port_trait( optical_to_thermal_homog, homography, "Homography" );
create_port_trait( thermal_to_optical_homog, homography, "Homography" );

//------------------------------------------------------------------------------
// Private implementation class
class itk_eo_ir_registration_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  bool m_output_frames_without_match;
  double m_max_time_offset;

  // Internal buffer
  struct entry
  {
    entry( kwiver::vital::image_container_sptr i,
           kwiver::vital::timestamp t,
           std::string name )
     : image( i ),
       time( t ),
       name( n )
    {}

    kwiver::vital::image_container_sptr image;
    kwiver::vital::timestamp time;
    std::string name;
  };

  std::list< entry > m_optical_frames;
  std::list< entry > m_thermal_frames;

  bool m_optical_finished;
  bool m_thermal_finished;

  // Helper functions
  bool attempt_registration(
    kwiver::vital::image_container_sptr optical,
    kwiver::vital::image_container_sptr thermal,
    kwiver::vital::homography& output );
};

// =============================================================================

itk_eo_ir_registration_process
::itk_eo_ir_registration_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new itk_eo_ir_registration_process::priv )
{
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}


itk_eo_ir_registration_process
::~itk_eo_ir_registration_process()
{
}


// -----------------------------------------------------------------------------
void
itk_eo_ir_registration_process
::_configure()
{
  d->m_output_frames_without_match =
    config_value_using_trait( output_frames_without_match );
  d->m_max_time_offset =
    config_value_using_trait( max_time_offset );
}


// -----------------------------------------------------------------------------
void
itk_eo_ir_registration_process
::_step()
{
  kwiver::vital::timestamp optical_time;
  kwiver::vital::image_container_sptr optical_image;
  std::string optical_file_name;

  kwiver::vital::timestamp thermal_time;
  kwiver::vital::image_container_sptr thermal_image;
  std::string thermal_file_name;

  if( has_input_port_edge_using_trait( optical_timestamp ) )
  {
    optical_time = grab_input_using_trait( optical_timestamp );

    LOG_DEBUG( logger(), "Received optical frame " << optical_time );
  }

  if( has_input_port_edge_using_trait( optical_image ) )
  {
    //optical_image = grab_input_using_trait( optical_image ); FIXME
  }

  if( has_input_port_edge_using_trait( optical_file_name ) )
  {
    optical_file_name = grab_input_using_trait( optical_file_name );
  }

  if( has_input_port_edge_using_trait( thermal_timestamp ) )
  {
    thermal_time = grab_input_using_trait( thermal_timestamp );

    LOG_DEBUG( logger(), "Received thermal frame " << thermal_time );
  }

  if( has_input_port_edge_using_trait( thermal_image ) )
  {
    //thermal_image = grab_input_using_trait( thermal_image ); FIXME
  }

  if( has_input_port_edge_using_trait( thermal_file_name ) )
  {
    thermal_file_name = grab_input_using_trait( thermal_file_name );
  }

  // Add images to buffer
  if( optical_image )
  {
    optical_frames.push_back(
      priv::entry( optical_image, optical_time, optical_file_name ) );
  }

  if( thermal_image )
  {
    thermal_frames.push_back(
      priv::entry( thermal_image, thermal_time, thermal_file_name ) );
  }

  // Determine if any images need to be tested
  bool optical_dominant = true;

  if( has_output_port_edge_using_trait( warped_optical_image ) )
  {
    optical_dominant = false;

    if( has_output_port_edge_using_trait( warped_optical_image ) )
    {
      throw std::runtime_error( "Cannot connect both warp image ports" );
    }
  }

}


// -----------------------------------------------------------------------------
void
itk_eo_ir_registration_process
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
itk_eo_ir_registration_process
::make_config()
{
  declare_config_using_trait( output_frames_without_match );
  declare_config_using_trait( max_time_offset );
}


// =============================================================================
itk_eo_ir_registration_process::priv
::priv()
{
}


itk_eo_ir_registration_process::priv
::~priv()
{
}

} // end namespace itk

} // end namespace viame
