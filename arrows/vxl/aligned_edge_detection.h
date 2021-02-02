// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VXL_ALIGNED_EDGE_DETECTION_
#define KWIVER_ARROWS_VXL_ALIGNED_EDGE_DETECTION_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {

namespace arrows {

namespace vxl {


/// Extract axis-aligned edges.
class KWIVER_ALGO_VXL_EXPORT aligned_edge_detection
  : public vital::algo::image_filter
{
public:
  PLUGIN_INFO( "vxl_aligned_edge_detection",
               "Compute axis-aligned edges in an image." )

  aligned_edge_detection();
  virtual ~aligned_edge_detection();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink.
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block.
  virtual void set_configuration( vital::config_block_sptr config );
  /// Check that the algorithm's current configuration is valid.
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Convert to the right type and optionally transform.
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
