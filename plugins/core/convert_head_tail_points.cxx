 /* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

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
