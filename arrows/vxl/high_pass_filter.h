// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VXL_HIGH_PASS_FILTER_
#define KWIVER_ARROWS_VXL_HIGH_PASS_FILTER_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace kwiver {

namespace arrows {

namespace vxl {

 /// VXL High Pass Filtering Process
 ///
 /// This method contains basic methods for high pass image filtering
 /// on top of input images
class KWIVER_ALGO_VXL_EXPORT high_pass_filter
  : public vital::algo::image_filter
{
public:
  PLUGIN_INFO( "vxl_high_pass_filter",
               "Use VXL to create an image based on high-frequency information." )

  high_pass_filter();
  virtual ~high_pass_filter();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  /// Perform high pass filtering
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
