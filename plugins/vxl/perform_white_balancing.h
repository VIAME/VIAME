/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_VXL_PERFORM_WHITE_BALANCING_H
#define VIAME_VXL_PERFORM_WHITE_BALANCING_H

#include <plugins/vxl/viame_vxl_export.h>

#include <vital/algo/image_filter.h>

namespace viame {

namespace kv = kwiver::vital;

/**
 * @brief VXL automatic white balancing process
 *
 * This method contains basic methods for image filtering on top of input
 * images via performing assorted white balancing operations.
 */
class VIAME_VXL_EXPORT perform_white_balancing
  : public kv::algorithm_impl< perform_white_balancing,
      kv::algo::image_filter >
{
public:

  PLUGIN_INFO( "vxl_white_balancing",
               "Perform auotomatic white balancing on some input." )

  perform_white_balancing();
  virtual ~perform_white_balancing();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kv::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kv::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kv::config_block_sptr config ) const;

  /// Perform white balancing
  virtual kv::image_container_sptr filter(
    kv::image_container_sptr image_data );

private:

  class priv;
  const std::unique_ptr<priv> d;
};

} // end namespace viame

#endif /* VIAME_VXL_PERFORM_WHITE_BALANCING_H */
