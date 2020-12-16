// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_CORE_VIDEO_INPUT_SPLIT_H
#define ARROWS_CORE_VIDEO_INPUT_SPLIT_H

#include <vital/algo/video_input.h>

#include <arrows/core/kwiver_algo_core_export.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Video input that pulls image and metadata inputs from different sources.
// ----------------------------------------------------------------
/**
 * This class implements a video input algorithm that holds two other video
 * input algorithms and pulls imagery from one and metadata from the other.
 */
class KWIVER_ALGO_CORE_EXPORT video_input_split
  : public  vital::algo::video_input
{
public:
  PLUGIN_INFO( "split",
               "Coordinate two video readers."
               " One reader supplies the image/data stream."
               " The other reader supplies the metadata stream." )

  /// Constructor
  video_input_split();
  virtual ~video_input_split();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual void open( std::string name );
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
  virtual kwiver::vital::image_container_sptr frame_image();
  virtual kwiver::vital::metadata_vector frame_metadata();
  virtual kwiver::vital::metadata_map_sptr metadata_map();

private:
  kwiver::vital::timestamp merge_timestamps(
    kwiver::vital::timestamp const& image_ts,
    kwiver::vital::timestamp const& metadata_ts ) const;

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif // ARROWS_CORE_VIDEO_INPUT_SPLIT_H
