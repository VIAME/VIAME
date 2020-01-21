/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
  if ( img.isNull() )
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

  if ( !img.isNull() )
  {
    img.save( qt_string( filename ) );
  }
}

} // end namespace qt

} // end namespace arrows

} // end namespace kwiver
