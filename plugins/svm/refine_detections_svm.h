/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Header for refine detections using SVM
 */

#ifndef VIAME_SVM_REFINE_DETECTIONS_SVM_H
#define VIAME_SVM_REFINE_DETECTIONS_SVM_H

#include "viame_svm_export.h"

#include <vital/algo/refine_detections.h>

namespace viame {

namespace kv = kwiver::vital;

/// A class for drawing various information about feature tracks
class VIAME_SVM_EXPORT refine_detections_svm
: public kv::algorithm_impl<refine_detections_svm,
    kv::algo::refine_detections>
{
public:

  PLUGIN_INFO( "svm_refiner",
               "Refine detections using SVM." )

  /// Constructor
  refine_detections_svm();

  /// Destructor
  virtual ~refine_detections_svm();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual kv::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(kv::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(kv::config_block_sptr config) const;

  /// Refine all object detections on the provided image
  /**
   * This method analyzes the supplied image and and detections on it,
   * returning a refined set of detections.
   *
   * \param image_data the image pixels
   * \param detections detected objects
   * \returns vector of image objects refined
   */
  virtual kv::detected_object_set_sptr
  refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const;

private:

  /// Private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace viame

#endif // VIAME_SVM_REFINE_DETECTIONS_SVM_H
