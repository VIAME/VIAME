// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief GDAL image_io interface
 */

#ifndef KWIVER_ARROWS_GDAL_IMAGE_IO_H_
#define KWIVER_ARROWS_GDAL_IMAGE_IO_H_

#include <arrows/gdal/kwiver_algo_gdal_export.h>

#include <vital/algo/image_io.h>

namespace kwiver {
namespace arrows {
namespace gdal {

/// A class for using GDAL to read and write images
class KWIVER_ALGO_GDAL_EXPORT image_io
  : public vital::algo::image_io
{
public:
  // No configuration for this class yet
  /// \cond DoxygenSuppress
  virtual void set_configuration(vital::config_block_sptr /*config*/) { }
  virtual bool check_configuration(vital::config_block_sptr /*config*/) const { return true; }
  /// \endcond

private:
  /// Implementation specific load functionality.
  /**
   * \param filename the path to the file the load
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
};

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver

#endif
