/*ckwg +30
 * Copyright 2017-2018, 2020 by Kitware, Inc.
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

#ifndef ARROWS_CORE_VIDEO_INPUT_IMAGE_LIST_H
#define ARROWS_CORE_VIDEO_INPUT_IMAGE_LIST_H

#include <vital/algo/video_input.h>

#include <arrows/core/kwiver_algo_core_export.h>

namespace kwiver {

namespace arrows {

namespace core {

// ----------------------------------------------------------------------------
/**
 * \brief Video input using list of images.
 *
 * This class implements a video input algorithm using a list of images
 * to simulate a video. Only the images are returned.
 * This algorithm produces no metadata.
 *
 * Example config:
 *   # select reader type
 *   image_reader:type = vxl
 */
class KWIVER_ALGO_CORE_EXPORT video_input_image_list
  : public vital::algo::video_input
{
public:
  PLUGIN_INFO( "image_list",
               "Read a list of images from a list of file names"
               " and presents them in the same way as reading a video."
               " The actual algorithm to read an image is specified"
               " in the \"image_reader\" config block."
               " Read an image list as a video stream." )

  video_input_image_list();
  virtual ~video_input_image_list();

  /**
   * \brief Get this algorithm's
   * \link vital::config_block configuration block \endlink.
   */
  vital::config_block_sptr get_configuration() const override;

  /**
   * \brief Set this algorithm's properties via a
   * \link vital::config_block configuration block \endlink.
   */
  void set_configuration( vital::config_block_sptr config ) override;

  /// Check that the algorithm's currently configuration is valid.
  bool check_configuration( vital::config_block_sptr config ) const override;

  /**
   * \brief Open a list of images.
   *
   * This method opens the file that contains the list of images. Each
   * image verified to exist at this time.
   *
   * \param list_name Name of file that contains list of images.
   */
  void open( std::string list_name ) override;
  void close() override;

  bool end_of_video() const override;
  bool good() const override;
  bool seekable() const override;
  size_t num_frames() const override;

  bool next_frame( kwiver::vital::timestamp& ts,
                   uint32_t timeout = 0 ) override;

  bool seek_frame( kwiver::vital::timestamp& ts,
                   kwiver::vital::timestamp::frame_t frame_number,
                   uint32_t timeout = 0 ) override;

  kwiver::vital::timestamp frame_timestamp() const override;
  kwiver::vital::image_container_sptr frame_image() override;
  kwiver::vital::metadata_vector frame_metadata() override;
  kwiver::vital::metadata_map_sptr metadata_map() override;

private:
  /**
   * \brief Private implementation class.
   */
  class priv;

  std::unique_ptr< priv > const d;
};

} // namespace core

} // namespace arrows

} // namespace kwiver

#endif
