/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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
 * \brief Qt image_io interface
 */

#ifndef KWIVER_ARROWS_QT_IMAGE_IO_H_
#define KWIVER_ARROWS_QT_IMAGE_IO_H_

#include <arrows/qt/kwiver_algo_qt_export.h>

#include <vital/algo/image_io.h>

namespace kwiver {
namespace arrows {
namespace qt {

/// A class for using Qt to read and write images.
///
/// This class provides an algorithm which can be used to read and write image
/// files using Qt. This algorithm is quite limited in terms of what formats
/// are supported, and offers no configuration. It is intended more as a proof
/// of concept.
class KWIVER_ALGO_QT_EXPORT image_io
  : public vital::algo::image_io
{
public:
  /// Constructor
  image_io();

  /// Destructor
  virtual ~image_io();

  PLUGIN_INFO( "qt",
               "Use Qt to load and save image files." )

  /// \copydoc vital::algo::image_io::set_configuration
  virtual void set_configuration(
    vital::config_block_sptr config ) override;
  /// \copydoc vital::algo::image_io::check_configuration
  virtual bool check_configuration(
    vital::config_block_sptr config ) const override;

private:
  /// \copydoc vital::algo::image_io::load_
  virtual vital::image_container_sptr load_(
    std::string const& filename ) const override;

  /// \copydoc vital::algo::image_io::save_
  virtual void save_( std::string const& filename,
                      vital::image_container_sptr data ) const override;
};

} // end namespace qt
} // end namespace arrows
} // end namespace kwiver

#endif
