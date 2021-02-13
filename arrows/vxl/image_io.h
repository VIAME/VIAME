// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL image_io interface
 */

#ifndef KWIVER_ARROWS_VXL_IMAGE_IO_H_
#define KWIVER_ARROWS_VXL_IMAGE_IO_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_io.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A class for using VXL to read and write images
class KWIVER_ALGO_VXL_EXPORT image_io
  : public vital::algo::image_io
{
public:
  PLUGIN_INFO( "vxl",
               "Use VXL (vil) to load and save image files." )

  /// Constructor
  image_io();

  /// Destructor
  virtual ~image_io();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;
  /// Get plane filenames for a given written file
  std::vector< std::string > plane_filenames( std::string const& filename ) const;

private:
  /// Implementation specific load functionality.
  /*
   * NOTE: When loading boolean images (ppm, pbm, etc.), true-value regions are
   * represented in the returned image as regions of 1's.
   *
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

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
