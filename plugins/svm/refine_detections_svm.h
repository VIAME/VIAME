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
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

namespace kv = kwiver::vital;

/// A class for drawing various information about feature tracks
class VIAME_SVM_EXPORT refine_detections_svm
  : public kv::algo::refine_detections
{
public:
#define VIAME_SVM_RDS_PARAMS \
    PARAM_DEFAULT( \
      model_dir, std::string, \
      "The directory where the SVM models are placed.", \
      "" ), \
    PARAM_DEFAULT( \
      override_original, bool, \
      "Replace original scores with new scores.", \
      true )

  PLUGGABLE_VARIABLES( VIAME_SVM_RDS_PARAMS )
  PLUGGABLE_CONSTRUCTOR( refine_detections_svm, VIAME_SVM_RDS_PARAMS )

  static std::string plugin_name() { return "svm_refiner"; }
  static std::string plugin_description() { return "Refine detections using SVM."; }

  PLUGGABLE_STATIC_FROM_CONFIG( refine_detections_svm, VIAME_SVM_RDS_PARAMS )
  PLUGGABLE_STATIC_GET_DEFAULT( VIAME_SVM_RDS_PARAMS )
  PLUGGABLE_SET_CONFIGURATION( refine_detections_svm, VIAME_SVM_RDS_PARAMS )

  /// Destructor
  virtual ~refine_detections_svm();

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
  void initialize() override;
  void set_configuration_internal( kv::config_block_sptr config ) override;

  /// Private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d_ );
};

} // end namespace viame

#endif // VIAME_SVM_REFINE_DETECTIONS_SVM_H
