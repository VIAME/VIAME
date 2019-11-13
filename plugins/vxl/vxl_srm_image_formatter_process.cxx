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
 * \brief Register multi-modal images.
 */

#include "vxl_srm_image_formatter_process.h"

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

namespace vxl
{

create_config_trait( fix_output_size, bool, "true",
  "Should the output image size always be consistent and unchanging" );

create_config_trait( max_image_width, unsigned, "1500",
  "Maximum allowed image width of archive after a potential resize" );
create_config_trait( max_image_height, unsigned, "1500",
  "Maximum allowed image height of archive after a potential resize" );

create_config_trait( resize_option, std::string, "rescale",
  "Option to meet output size parameter, can be: rescale, chip, or crop." );

create_config_trait( chip_overlap, unsigned, "50",
  "If we're chipping a large image into smaller chips, this is the approximate "
  "overlap between neighboring chips in terms of pixels." );
create_config_trait( flux_factor, double, "0.05",
  "Allowable error for resizing images to meet a more desirable size." );



//------------------------------------------------------------------------------
// Private implementation class
class vxl_srm_image_formatter_process::priv
{
public:
  priv();
  ~priv();

  bool m_fix_output_size;

  unsigned m_max_image_width;
  unsigned m_max_image_height;

  enum{ RESCALE, CHIP, CROP } m_resize_option;

  unsigned m_chip_overlap;
  double m_flux_factor;
};

// =============================================================================

vxl_srm_image_formatter_process
::vxl_srm_image_formatter_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new vxl_srm_image_formatter_process::priv() )
{
  make_ports();
  make_config();
}


vxl_srm_image_formatter_process
::~vxl_srm_image_formatter_process()
{
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
::_configure()
{
  d->m_fix_output_size =
    config_value_using_trait( fix_output_size );
  d->m_max_image_width =
    config_value_using_trait( max_image_width );
  d->m_max_image_height =
    config_value_using_trait( max_image_height );
  d->m_chip_overlap =
    config_value_using_trait( chip_overlap );
  d->m_flux_factor =
    config_value_using_trait( flux_factor );

  std::string mode = config_value_using_trait( resize_option );

  if( mode == "rescale" )
  {
    d->m_resize_option = priv::RESCALE;
  }
  else if( mode == "chip" )
  {
    d->m_resize_option = priv::CHIP;
  }
  else if( mode == "crop" )
  {
    d->m_resize_option = priv::CROP;
  }
  else
  {
    throw std::runtime_error( "Unable to identify resize option value: " + mode );
  }
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
::_step()
{
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( image, optional );
}


// -----------------------------------------------------------------------------
void
vxl_srm_image_formatter_process
::make_config()
{
  declare_config_using_trait( fix_output_size );
  declare_config_using_trait( max_image_width );
  declare_config_using_trait( max_image_height );
  declare_config_using_trait( resize_option );
  declare_config_using_trait( chip_overlap );
  declare_config_using_trait( flux_factor );
}


// =============================================================================
vxl_srm_image_formatter_process::priv
::priv()
{
}


vxl_srm_image_formatter_process::priv
::~priv()
{
}


} // end namespace vxl

} // end namespace viame
