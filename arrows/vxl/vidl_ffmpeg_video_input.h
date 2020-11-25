// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for video input using VXL methods.
 */

#ifndef KWIVER_ARROWS_VXL_VIDL_FFMPEG_VIDEO_INPUT_H
#define KWIVER_ARROWS_VXL_VIDL_FFMPEG_VIDEO_INPUT_H

#include <vital/algo/video_input.h>

#include <arrows/vxl/kwiver_algo_vxl_export.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// Video input using VXL vidl ffmpeg services.
// ----------------------------------------------------------------
/**
 * This class implements a video input algorithm using the VXL vidl
 * ffmpeg video services.
 *
 */
class KWIVER_ALGO_VXL_EXPORT vidl_ffmpeg_video_input
  : public vital::algo::video_input
{
public:
  PLUGIN_INFO( "vidl_ffmpeg",
               "Use VXL (vidl with FFMPEG) to read video files as a sequence of images." )

  /// Constructor
  vidl_ffmpeg_video_input();
  virtual ~vidl_ffmpeg_video_input();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual void open( std::string video_name );
  virtual void close();

  virtual bool end_of_video() const;
  virtual bool good() const;
  virtual bool seekable() const;
  virtual size_t num_frames() const;

  virtual bool next_frame( kwiver::vital::timestamp& ts,
                           uint32_t timeout = 0 );

  virtual bool seek_frame( kwiver::vital::timestamp& ts,
                           kwiver::vital::timestamp::frame_t frame_number,
                           uint32_t timeout = 0 );

  virtual kwiver::vital::timestamp frame_timestamp() const;

  virtual double frame_rate();

  virtual kwiver::vital::image_container_sptr frame_image();
  virtual kwiver::vital::metadata_vector frame_metadata();
  virtual kwiver::vital::metadata_map_sptr metadata_map();

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_VXL_VIDL_FFMPEG_VIDEO_INPUT_H
