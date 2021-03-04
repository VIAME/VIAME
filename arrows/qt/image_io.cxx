// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Qt image_io implementation
 */

#include "image_io.h"

#include <arrows/qt/image_container.h>

namespace {

// ----------------------------------------------------------------------------
QString
qt_string( std::string const& in )
{
  return QString::fromLocal8Bit( in.data(), static_cast< int >( in.size() ) );
}

} // end namespace (anonymous)

namespace kwiver {

namespace arrows {

namespace qt {

// ----------------------------------------------------------------------------
image_io
::image_io()
{
  attach_logger( "arrows.qt.image_io" );
}

// ----------------------------------------------------------------------------
image_io
::~image_io()
{
}

// ----------------------------------------------------------------------------
void
image_io
::set_configuration( vital::config_block_sptr in_config )
{
  static_cast< void >( in_config );
}

// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
image_io
::check_configuration( vital::config_block_sptr config ) const
{
  static_cast< void >( config );
  return true;
}

// ----------------------------------------------------------------------------
// Load image image from the file
vital::image_container_sptr
image_io
::load_( std::string const& filename ) const
{
  LOG_DEBUG( logger(), "Loading image from file: " << filename );

  auto img = QImage{ qt_string( filename ) };
  if( img.isNull() )
  {
    return {};
  }

  return std::make_shared< image_container >( img );
}

// ----------------------------------------------------------------------------
void
image_io
::save_( std::string const& filename, vital::image_container_sptr data ) const
{
  auto qdata = std::dynamic_pointer_cast< image_container >( data );
  auto const& img =
    ( qdata ? ( *qdata ) : image_container::vital_to_qt( data->get_image() ) );

  if( !img.isNull() )
  {
    img.save( qt_string( filename ) );
  }
}

} // end namespace qt

} // end namespace arrows

} // end namespace kwiver
