// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief \todo
 */

#ifndef KWIVER_ARROWS_FFMPEG_FFMPEG_VIDEO_INPUT_H
#define KWIVER_ARROWS_FFMPEG_FFMPEG_VIDEO_INPUT_H

#include <vital/algo/video_input.h>

#include <arrows/ffmpeg/kwiver_algo_ffmpeg_export.h>

namespace kwiver {
namespace arrows {
namespace ffmpeg {

/// Video input using ffmpeg services.
// ---------------------------------------------------------------------------
/**
 * This class implements a video input algorithm using ffmpeg video services.
 *
 */
class KWIVER_ALGO_FFMPEG_EXPORT ffmpeg_video_input
  : public  vital::algo::video_input
{
public:
  /// Constructor
  ffmpeg_video_input();
  virtual ~ffmpeg_video_input();

  PLUGIN_INFO( "ffmpeg",
               "Use FFMPEG to read video files as a sequence of images." )

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  vital::config_block_sptr get_configuration() const override;

  /// Set this algorithm's properties via a config block
  void set_configuration(vital::config_block_sptr config) override;

  /// Check that the algorithm's currently configuration is valid
  bool check_configuration(vital::config_block_sptr config) const override;

  void open( std::string video_name ) override;
  void close() override;

  bool end_of_video() const override;
  bool good() const override;

  bool seekable() const override;
  size_t num_frames() const override;

  bool next_frame( ::kwiver::vital::timestamp& ts,
                   uint32_t timeout = 0 ) override;
  bool seek_frame( ::kwiver::vital::timestamp& ts,
                   ::kwiver::vital::timestamp::frame_t frame_number,
                   uint32_t timeout = 0) override;

  ::kwiver::vital::timestamp frame_timestamp() const override;
  ::kwiver::vital::image_container_sptr frame_image() override ;
  ::kwiver::vital::metadata_vector frame_metadata() override;
  ::kwiver::vital::metadata_map_sptr metadata_map() override;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_FFMPEG_FFMPEG_VIDEO_INPUT_H
