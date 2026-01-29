// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VIAME_CORE_MERGE_DETECTIONS_SUPPRESS_IN_REGIONS_H
#define VIAME_CORE_MERGE_DETECTIONS_SUPPRESS_IN_REGIONS_H

#include "viame_core_export.h"

#include <vital/algo/merge_detections.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * \class merge_detections_suppress_in_regions
 *
 * \brief Prunes detections overlapping with regions identified by class string
 */
class VIAME_CORE_EXPORT merge_detections_suppress_in_regions
  : public kwiver::vital::algo::merge_detections
{

public:
  PLUGGABLE_IMPL(
    merge_detections_suppress_in_regions,
    "Suppresses detections within regions indicated by a certain fixed category "
    "of detections. Can either remove the detections or reduce their probability.",
    PARAM_DEFAULT(
      suppression_class, std::string,
      "Suppression region class IDs, will eliminate any detections overlapping with "
      "this class entirely.",
      "" ),
    PARAM_DEFAULT(
      borderline_class, std::string,
      "Borderline region class IDs, will reduce the probability of any detections "
      "overlapping with the class by some fixed scale factor.",
      "" ),
    PARAM_DEFAULT(
      borderline_scale_factor, double,
      "The factor by which the detections are scaled when overlapping with borderline "
      "regions.",
      0.5 ),
    PARAM_DEFAULT(
      min_overlap, double,
      "The minimum percent a detection can overlap with a suppression category before "
      "it's discarded or reduced. Range [0.0,1.0].",
      0.5 ),
    PARAM_DEFAULT(
      output_region_classes, bool,
      "Add suppression and borderline classes to output.",
      true ),
    PARAM_DEFAULT(
      case_sensitive, bool,
      "Treat class names as case sensitive or insensitive.",
      false )
  )

  virtual ~merge_detections_suppress_in_regions() = default;

  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Refine all input object detections
  /**
   * This method suppresses detections given by ID
   *
   * \param sets Input detection sets
   * \returns vector of refined detections
   */
  virtual kwiver::vital::detected_object_set_sptr
  merge( std::vector< kwiver::vital::detected_object_set_sptr > const& sets ) const;

private:
  bool compare_classes( const std::string& c1, const std::string& c2 ) const;

}; // end class merge_detections_suppress_in_regions

} // end namespace viame

#endif // VIAME_CORE_MERGE_DETECTIONS_SUPPRESS_IN_REGIONS_H
