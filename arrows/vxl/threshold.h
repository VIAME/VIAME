// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_THRESHOLD_
#define KWIVER_THRESHOLD_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {

namespace arrows {

namespace vxl {

/// Threshold an image using different schemes.
///
/// Use either an absolute threshold or one based on percentiles.
class KWIVER_ALGO_VXL_EXPORT threshold
  : public vital::algo::image_filter
{
public:
  PLUGIN_INFO( "vxl_threshold",
               "Threshold at image at a given percentile or value." )

  threshold();
  virtual ~threshold();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink.
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block.
  virtual void set_configuration( vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid.
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Binarize the image at a given percentile threshold.
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
