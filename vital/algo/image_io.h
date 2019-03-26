/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief Interface for image_io \link kwiver::vital::algo::algorithm_def algorithm
 *        definition \endlink.
 */

#ifndef VITAL_ALGO_IMAGE_IO_H_
#define VITAL_ALGO_IMAGE_IO_H_

#include <vital/vital_config.h>

#include <string>

#include <vital/algo/algorithm.h>
#include <vital/algorithm_capabilities.h>
#include <vital/types/image_container.h>
#include <vital/types/metadata.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for reading and writing images
/**
 * This class represents an abstract interface for reading and writing
 * images.
 *
 * A note about the basic capabilities:
 *
 * HAS_TIME - This capability is set to true if the image metadata
 *     supplies a timestamp. If a timestamp is supplied, it is made
 *     available in the metadata for the image. If the timestamp
 *     is not supplied, then the metadata will not have the timestamp set.
 */
class VITAL_ALGO_EXPORT image_io
  : public kwiver::vital::algorithm_def<image_io>
{
public:
  // Common capabilities
  // -- basic capabilities --
  static const algorithm_capabilities::capability_name_t HAS_TIME;

  virtual ~image_io() = default;

  /// Return the name of this algorithm
  static std::string static_type_name() { return "image_io"; }

  /// Load image from the file
  /**
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   *
   * \param filename the path to the file to load
   * \returns an image container refering to the loaded image
   */
  kwiver::vital::image_container_sptr load(std::string const& filename) const;

  /// Save image to a file
  /**
   * Image file format is based on file extension.
   *
   * \throws kwiver::vital::path_not_exists Thrown when the expected
   *    containing directory of the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_directory Thrown when the expected
   *    containing directory of the given path is not actually a
   *    directory.
   *
   * \param filename the path to the file to save
   * \param data the image container refering to the image to write
   */
  void save(std::string const& filename, kwiver::vital::image_container_sptr data) const;

  /// Get the image metadata
  /**
   * \throws kwiver::vital::path_not_exists Thrown when the given path does not exist.
   *
   * \throws kwiver::vital::path_not_a_file Thrown when the given path does
   *    not point to a file (i.e. it points to a directory).
   *
   * \param filename the path to the file to read
   * \returns pointer to the loaded metadata
   */
  kwiver::vital::metadata_sptr load_metadata(std::string const& filename) const;

  /**
   * \brief Return capabilities of concrete implementation.
   *
   * This method returns the capabilities for the current image reader/writer.
   *
   * \return Reference to supported image capabilities.
   */
  algorithm_capabilities const& get_implementation_capabilities() const;

protected:
  image_io();

  void set_capability( algorithm_capabilities::capability_name_t const& name, bool val );

private:
  /// Implementation specific load functionality.
  /**
   * Concrete implementations of image_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file the load
   * \returns an image container refering to the loaded image
   */
  virtual kwiver::vital::image_container_sptr load_(std::string const& filename) const = 0;

  /// Implementation specific save functionality.
  /**
   * Concrete implementations of image_io class must provide an
   * implementation for this method.
   *
   * \param filename the path to the file to save
   * \param data the image container refering to the image to write
   */
  virtual void save_(std::string const& filename,
                     kwiver::vital::image_container_sptr data) const = 0;

  /// Implementation specific metadata functionality.
  /**
   * If a concrete implementation provides metadata, it must be provided
   * in both load() and load_metadata(), and it must be the same metadata.
   * To provide it in one but not the other, or to provide different metadata
   * in each, is an error.
   *
   * \param filename the path to the file to read
   * \returns pointer to the loaded metadata
   */
  virtual kwiver::vital::metadata_sptr load_metadata_(std::string const& filename) const;

  algorithm_capabilities m_capabilities;
};


/// Shared pointer type for generic image_io definition type.
typedef std::shared_ptr<image_io> image_io_sptr;


} } } // end namespace

#endif // VITAL_ALGO_IMAGE_IO_H_
