// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OpenCV image_io interface
 */

#ifndef KWIVER_ARROWS_OCV_IMAGE_IO_H_
#define KWIVER_ARROWS_OCV_IMAGE_IO_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/image_io.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A class for using OpenCV to read and write images
class KWIVER_ALGO_OCV_EXPORT image_io
  : public vital::algo::image_io
{
public:
  PLUGIN_INFO( "ocv",
               "Read and write image using OpenCV." )

  // No configuration for this class yet
  /// \cond DoxygenSuppress
  virtual void set_configuration(vital::config_block_sptr /*config*/) { }
  virtual bool check_configuration(vital::config_block_sptr /*config*/) const { return true; }
  /// \endcond

private:
  /// Implementation specific load functionality.
  /**
   * \param filename the path to the file to load
   * \returns an image container refering to the loaded image
   */
  virtual vital::image_container_sptr load_(const std::string& filename) const;

  /// Implementation specific save functionality.
  /**
   * \param filename the path to the file to save
   * \param data the image container refering to the image to write
   */
  virtual void save_(const std::string& filename,
                     vital::image_container_sptr data) const;

  /// Implementation specific metadata functionality.
  /**
   * \param filename the path to the file to read
   * \returns pointer to the loaded metadata
   */
  virtual kwiver::vital::metadata_sptr load_metadata_(std::string const& filename) const;
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
