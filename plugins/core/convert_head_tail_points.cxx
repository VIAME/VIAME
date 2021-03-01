 /*ckwg +29
 * Copyright 2021 by Kitware, Inc.
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

#include "convert_head_tail_points.h"

#include <cmath>
#include <string>

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class convert_head_tail_points::priv
{
public:

  priv()
   : m_head_postfix( "_head" )
   , m_tail_postfix( "_tail" )
   , m_box_expansion( 0.05 )
  {}

  ~priv() {}

  std::string m_head_postfix;
  std::string m_tail_postfix;
  double m_box_expansion;
};

// =============================================================================

convert_head_tail_points
::convert_head_tail_points()
  : d( new priv )
{}


convert_head_tail_points
::  ~convert_head_tail_points()
{}


// -----------------------------------------------------------------------------
kwiver::vital::config_block_sptr
convert_head_tail_points
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config =
    kwiver::vital::algorithm::get_configuration();

  config->set_value( "head_postfix", d->m_head_postfix,
    "Detection type postfix indicating head position." );
  config->set_value( "tail_postfix", d->m_tail_postfix,
    "Detection type postfix indicating tail position." );

  return config;
}


// -----------------------------------------------------------------------------
void
convert_head_tail_points
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_head_postfix = config->get_value< std::string >( "head_postfix" );
  d->m_tail_postfix = config->get_value< std::string >( "tail_postfix" );
}


// -----------------------------------------------------------------------------
bool
convert_head_tail_points
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
convert_head_tail_points
::refine( kwiver::vital::image_container_sptr image_data,
  kwiver::vital::detected_object_set_sptr input_dets ) const
{
  auto output = std::make_shared< kwiver::vital::detected_object_set >();

  return output;
}

} // end namespace
