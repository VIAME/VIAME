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

#ifndef VIAME_CORE_AUTO_DETECT_TRANSFORM_H
#define VIAME_CORE_AUTO_DETECT_TRANSFORM_H

#include <plugins/core/viame_core_export.h>

#include <vital/algo/transform_2d_io.h>
#include <vital/types/transform_2d.h>


namespace viame
{

/// Automatically detect transform type and load it
class VIAME_CORE_EXPORT auto_detect_transform_io
  : public kwiver::vital::algo::transform_2d_io
{
public:

  static constexpr char const* name = "auto";

  static constexpr char const* description = "Automatically detect a transform "
    "type stored in either an ITK (.h5) or simple homography (.txt) format.";

  /// Constructor
  auto_detect_transform_io();

  /// Destructor
  virtual ~auto_detect_transform_io();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual kwiver::vital::transform_2d_sptr load_(
    std::string const& filename ) const;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of transform_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::transform_2d_sptr data ) const;

};

} // end namespace viame

#endif // VIAME_CORE_AUTO_DETECT_TRANSFORM_H
