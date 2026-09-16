// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation for example_detector

#include "example_detector.h"

#include <viame/algorithm_framework/viame_compiler_config.h>

namespace viame {

namespace core {

class example_detector::priv
{
public:
  priv( example_detector& parent )
    : parent( parent ),
      m_frame_ct( 0 )
  {}

  example_detector& parent;

  // Configuration values
  double c_center_x() { return parent.c_center_x; }
  double c_center_y() { return parent.c_center_y; }
  double c_height() { return parent.c_height; }
  double c_width() { return parent.c_width; }
  double c_dx() { return parent.c_dx; }
  double c_dy() { return parent.c_dy; }

  ~priv()
  {}

  // Local value
  int m_frame_ct;
}; // end class example_detector::priv

// ----------------------------------------------------------------------------
void
example_detector
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "arrows.core.example_detector" );
}

example_detector::
~example_detector()
{}

// ----------------------------------------------------------------------------
bool
example_detector
::check_configuration( [[maybe_unused]] viame::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
viame::detected_object_set_sptr
example_detector
::detect( [[maybe_unused]] viame::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< viame::detected_object_set >();
  const double ct = static_cast< double >( d->m_frame_ct );

  viame::bounding_box_d bbox(
    d->c_center_x() + ct * d->c_dx() - d->c_width() / 2.0,
    d->c_center_y() + ct * d->c_dy() - d->c_height() / 2.0,
    d->c_center_x() + ct * d->c_dx() + d->c_width() / 2.0,
    d->c_center_y() + ct * d->c_dy() + d->c_height() / 2.0 );

  ++d->m_frame_ct;

  auto dot = std::make_shared< viame::detected_object_type >();
  dot->set_score( "detection", 1.0 );

  detected_set->add(
    std::make_shared< viame::detected_object >(
      bbox,
      1.0, dot ) );

  return detected_set;
}

} // namespace core

} // namespace viame
