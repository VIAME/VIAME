/*ckwg +30
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

/**
 * \file
 * \brief Interface for transform_2d_io
 *        \link kwiver::vital::algo::algorithm_def algorithm definition
 *        \endlink.
 */

#ifndef VITAL_ALGO_TRANSFORM_2D_IO_H_
#define VITAL_ALGO_TRANSFORM_2D_IO_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/types/transform_2d.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for reading and writing transforms
/**
 * This class represents an abstract interface for reading and writing
 * transforms.
 */
class VITAL_ALGO_EXPORT transform_2d_io
  : public kwiver::vital::algorithm_def<transform_2d_io>
{
public:
  virtual ~transform_2d_io() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "transform_2d_io"; }

  /// Load transform from the file
  /**
   * \throws kwiver::vital::path_not_exists
   *   Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file
   *   Thrown when the given path does not point to a file (i.e. it points to a
   *   directory).
   *
   * \param filename the path to the file to load
   * \returns a transform instance referring to the loaded transform
   */
  kwiver::vital::transform_2d_sptr load( std::string const& filename ) const;

  /// Save transform to a file
  /**
   * Transform file format is based on the algorithm instance.
   *
   * \throws kwiver::vital::path_not_exists
   *   Thrown when the expected containing directory of the given path does not
   *   exist.
   *
   * \throws kwiver::vital::path_not_a_directory
   *   Thrown when the expected containing directory of the given path is not
   *   actually a directory.
   *
   * \throws kwiver::vital::invalid_data
   *   Thrown when the algorithm does not recognize the concrete type of the
   *   transformation instance.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  void save( std::string const& filename,
             kwiver::vital::transform_2d_sptr data ) const;

protected:
  transform_2d_io();

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of transform_2d_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns a transform instance referring to the loaded transform
   */
  virtual kwiver::vital::transform_2d_sptr load_(
    std::string const& filename ) const = 0;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of transform_2d_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the transform instance referring to the transform to write
   */
  virtual void save_( std::string const& filename,
                      kwiver::vital::transform_2d_sptr data ) const = 0;
};

/// Shared pointer type for generic transform_2d_io definition type.
using transform_2d_io_sptr = std::shared_ptr< transform_2d_io >;

} // namespace algo
} // namespace vital
} // namespace kwiver

#endif
