/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_REFINE_DETECTIONS_ADD_FIXED_H
#define VIAME_CORE_REFINE_DETECTIONS_ADD_FIXED_H

#include "viame_core_export.h"

#include <vital/algo/refine_detections.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * \class refine_detections_add_fixed
 *
 * \brief Prunes overlapping detections
 *
 * \iports
 * \iport{detections}
 *
 * \oports
 * \oport{pruned_detections}
 */
class VIAME_CORE_EXPORT refine_detections_add_fixed
  : public kwiver::vital::algorithm_impl<refine_detections_add_fixed,
    kwiver::vital::algo::refine_detections>
{

public:

  PLUGIN_INFO( "add_fixed",
    "Adds a fixed detection into the current set.\n\n"
    "The fixed detection can be either a config defined box or "
    "based on the input image size." )

  refine_detections_add_fixed();
  virtual ~refine_detections_add_fixed();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(kwiver::vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(kwiver::vital::config_block_sptr config) const;

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual kwiver::vital::detected_object_set_sptr
  refine( kwiver::vital::image_container_sptr image_data,
          kwiver::vital::detected_object_set_sptr detections ) const;

private:

  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;

 }; // end class refine_detections_add_fixed


} // end namespace viame

#endif // VIAME_CORE_REFINE_DETECTIONS_ADD_FIXED_H
