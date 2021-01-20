// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VXL_CONVERT_IMAGE_
#define KWIVER_ARROWS_VXL_CONVERT_IMAGE_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {

namespace arrows {

namespace vxl {

/**
 * @brief Convert between VXL image formats.
 *
 * This can be used, for example, to turn a floating point image into
 * a byte image and vice versa.
 */
class KWIVER_ALGO_VXL_EXPORT convert_image
  : public vital::algo::image_filter
{
public:
  PLUGIN_INFO( "vxl_convert_image",
               "Convert image between different formats or scales." )

  convert_image();
  virtual ~convert_image();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Convert to the right type and optionally transform
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  class priv;

  const std::unique_ptr< priv > d;
};

} // namespace vxl

} // namespace arrows

} // namespace kwiver

#endif
