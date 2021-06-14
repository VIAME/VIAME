// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VXL_PIXEL_FEATURE_EXTRACTOR_
#define KWIVER_ARROWS_VXL_PIXEL_FEATURE_EXTRACTOR_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {

namespace arrows {

namespace vxl {

/// Extract multiple features from an image
class KWIVER_ALGO_VXL_EXPORT pixel_feature_extractor
  : public vital::algo::image_filter
{
public:
  PLUGIN_INFO( "vxl_pixel_feature_extractor",
               "Extract various local pixel-wise features from an image." )

  pixel_feature_extractor();
  virtual ~pixel_feature_extractor();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// extract local pixel-wise features
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  class priv;

  std::unique_ptr< priv > const d;
};

} // namespace vxl

} // namespace arrows

} // namespace kwiver

#endif
