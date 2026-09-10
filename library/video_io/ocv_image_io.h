// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OpenCV image_io interface

#ifndef KWIVER_ARROWS_OCV_IMAGE_IO_H_
#define KWIVER_ARROWS_OCV_IMAGE_IO_H_

#include "viame_video_io_export.h"

#include <viame/algorithm_framework/algo/image_io.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// A class for using OpenCV to read and write images
class VIAME_VIDEO_IO_EXPORT image_io
  : public vital::algo::image_io
{
public:
  PLUGGABLE_IMPL(
    image_io,
    "Read and write image using OpenCV." )

  // No configuration for this class yet
  /// \cond DoxygenSuppress
  virtual bool
  check_configuration( vital::config_block_sptr /*config*/ ) const
  {
    return true;
  }

  /// \endcond

private:
  /// Implementation specific load functionality.
  ///
  /// \param filename the path to the file to load
  /// \returns an image container refering to the loaded image
  virtual vital::image_container_sptr load_(
    const std::string& filename ) const;

  /// Implementation specific save functionality.
  ///
  /// \param filename the path to the file to save
  /// \param data the image container refering to the image to write
  virtual void save_(
    const std::string& filename,
    vital::image_container_sptr data ) const;

  /// Implementation specific metadata functionality.
  ///
  /// \param filename the path to the file to read
  /// \returns pointer to the loaded metadata
  virtual kwiver::vital::metadata_sptr load_metadata_(
    std::string const& filename ) const;
};

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver

#endif
