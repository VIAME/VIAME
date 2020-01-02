/*ckwg +29
 * Copyright 2019-2020 by Kitware, Inc.
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

#ifndef VIAME_ITK_ITK_TRANSFORM_H
#define VIAME_ITK_ITK_TRANSFORM_H

#include <plugins/itk/viame_itk_export.h>

#include <vital/algo/transform_2d_io.h>
#include <vital/types/transform_2d.h>

#include "RegisterOpticalAndThermal.h"

namespace viame
{

namespace itk
{

/// Wraps ITK transforms in KWIVER vital types
class VIAME_ITK_EXPORT ITKTransform : public kwiver::vital::transform_2d
{
public:

  ITKTransform( BaseTransformType::Pointer transform );
  virtual ~ITKTransform();

  /// Create a clone of this transform object, returning as smart pointer
  /**
   * \return A new deep clone of this transformation.
   */
  virtual kwiver::vital::transform_2d_sptr clone() const;

  /// Return an inverse of this transform object
  /**
   * \throws non_invertible_transform when the transformation is non-invertible.
   * \return A new transform object that is the inverse of this transformation.
   */
  virtual kwiver::vital::transform_2d_sptr inverse() const;

  /// Map a 2D double-type point using this transform
  /**
   * \param p Point to map against this transform
   * \return New point in the projected coordinate system.
   */
  virtual kwiver::vital::vector_2d map( kwiver::vital::vector_2d const& p ) const;

private:

  BaseTransformType::Pointer m_transform;
};


/// A class for using ITK to read and write arbitrary transforms
class VIAME_ITK_EXPORT ITKTransformIO
  : public kwiver::vital::algorithm_impl<
      ITKTransformIO, kwiver::vital::algo::transform_2d_io>
{
public:
  /// Constructor
  ITKTransformIO();

  /// Destructor
  virtual ~ITKTransformIO();

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


} // end namespace itk

} // end namespace viame

#endif // VIAME_ITK_ITK_TRANSFORM_H
