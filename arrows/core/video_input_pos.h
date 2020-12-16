// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_CORE_VIDEO_INPUT_POS_H
#define ARROWS_CORE_VIDEO_INPUT_POS_H

#include <vital/algo/video_input.h>

#include <arrows/core/kwiver_algo_core_export.h>

namespace kwiver {
namespace arrows {
namespace core {

/// Metadata reader using the AFRL POS file format.
// ----------------------------------------------------------------
/**
 * This class implements a video input algorithm that returns only metadata.
 *
 * The algorithm takes configuration for a directory full of images
 * and an associated directory name for the metadata files. These
 * metadata files have the same base name as the image files.
 */
class KWIVER_ALGO_CORE_EXPORT video_input_pos
  : public  vital::algo::video_input
{
public:
  PLUGIN_INFO(  "pos",
                "Read video metadata in AFRL POS format."
                " The algorithm takes configuration for a directory full of images"
                " and an associated directory name for the metadata files."
                " These metadata files have the same base name as the image files."
                " Each metadata file is associated with the image file"
                " of the same base name." )

  /// Constructor
  video_input_pos();
  virtual ~video_input_pos();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;

  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /**
   * @brief Open a list of images.
   *
   * This method opens the file that contains the list of images. The
   * individual image names are used to find the associated metadata
   * file in the directory supplied via the configuration.
   *
   * @param list_name Name of file that contains list of images.
   */
  virtual void open( std::string list_name );
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
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif // ARROWS_CORE_VIDEO_INPUT_POS_H
