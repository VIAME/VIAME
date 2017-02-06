/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#ifndef ARROWS_CORE_INS_DATA_READER_H
#define ARROWS_CORE_INS_DATA_READER_H

#include <vital/algo/video_input.h>

#include <arrows/core/kwiver_algo_core_export.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Metadata reader using ins/pos format files.
// ----------------------------------------------------------------
/**
 * This class implements a video input algorithm that returns only metadata.
 *
 * The algorithm takes configuration for a directory full of images
 * and an associated directory name for the metadata files. These
 * metadata files have the same base name as the image files.
 */
class KWIVER_ALGO_CORE_EXPORT ins_data_reader
  : public vital::algorithm_impl < ins_data_reader, vital::algo::video_input >
{
public:
  /// Constructor
  ins_data_reader();
  virtual ~ins_data_reader();

  /// copy constructor
  ins_data_reader( ins_data_reader const& other );

  /// Return the name of this implementation
  virtual std::string impl_name() const { return "image_list"; }

  virtual std::string description() const
  {
    return "Read image metadata in AFRL ins format. "
      "The algorithm takes configuration for a directory full of images "
      "and an associated directory name for the metadata files. These "
      "metadata files have the same base name as the image files. "
      "Each metadata file is associated with the image file."
      ;
  }

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /**
   * @brief Open a list of images.
   *
   * This method opens the file that contains the list of images. Each
   * image verified to exist at this time.
   *
   * @param list_name Name of file that contains list of images.
   */
  virtual void open( std::string list_name );
  virtual void close();

  virtual bool end_of_video() const;
  virtual bool good() const;

  virtual bool next_frame( kwiver::vital::timestamp& ts,
                           uint32_t timeout = 0 );

  virtual kwiver::vital::image_container_sptr frame_image();
  virtual kwiver::vital::video_metadata_vector frame_metadata();

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif /* ARROWS_CORE_INS_DATA_READER_H */
