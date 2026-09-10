// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CLASS_PROBABILITY_FILTER_H_
#define KWIVER_ARROWS_CLASS_PROBABILITY_FILTER_H_

#include "viame_classifiers_export.h"

#include <viame/algorithm_framework/algo/detected_object_filter.h>

#include <viame/algorithm_framework/algo/algorithm.txx>

#include <set>
#include <utility>

namespace kwiver {

namespace arrows {

namespace core {

// ----------------------------------------------------------------------------
/// @brief Filters detections based on class probability.
///
/// This algorithm filters out items that are less than the threshold.
/// The following steps are applied to each input detected object set.
///
/// 1) Select all class names with scores greater than threshold.
///
/// 2) Create a new detected_object_type object with all selected class
///    names from step 1. The class name can be selected individually
///    or with the keep_all_classes option.
///
/// 3) The input detection_set is cloned and the detected_object_type
///    from step 2 is attached.

class VIAME_CLASSIFIERS_EXPORT class_probability_filter
  : public vital::algo::detected_object_filter
{
public:
  PLUGGABLE_IMPL(
    class_probability_filter,
    "Filters detections based on class probability.\n\n"
    "This algorithm filters out items that are less than the threshold."
    " The following steps are applied to each input detected object set.\n\n"
    "1) Select all class names with scores greater than threshold.\n\n"
    "2) Create a new detected_object_type object with all selected class"
    " names from step 1. The class name can be selected individually"
    " or with the keep_all_classes option.\n\n"
    "3) The input detection_set is cloned and the detected_object_type"
    " from step 2 is attached.",
    PARAM_DEFAULT(
      threshold, double,
      "Detections are passed through this filter if they have a selected classification that is "
      "above this threshold.",
      0.0 ),
    PARAM_DEFAULT(
      keep_all_classes, bool,
      "If this options is set to true, all classes are passed through this filter "
      "if they are above the selected threshold.",
      true ),
    PARAM_DEFAULT(
      list_of_classes,
      std::string,
      "A list of class names to pass through this filter. "
      "Multiple names are separated by a ';' character. "
      "The keep_all_classes parameter overrides this list of classes. "
      "So be sure to set that to false if you only want the listed classes.",
      "" )
  )

  virtual ~class_probability_filter() = default;

  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual vital::detected_object_set_sptr filter(
    const vital::detected_object_set_sptr input_set ) const;
};

} // namespace core

} // namespace arrows

}   // End namespace

#endif
