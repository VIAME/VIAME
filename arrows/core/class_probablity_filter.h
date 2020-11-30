// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CLASS_PROBABLITY_FILTER_H_
#define KWIVER_ARROWS_CLASS_PROBABLITY_FILTER_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/detected_object_filter.h>

#include <utility>
#include <set>

namespace kwiver {
namespace arrows {
namespace core {

// ----------------------------------------------------------------
/**
 * @brief Filters detections based on class probability.
 *
 * This algorithm filters out items that are less than the threshold.
 * The following steps are applied to each input detected object set.
 *
 * 1) Select all class names with scores greater than threshold.
 *
 * 2) Create a new detected_object_type object with all selected class
 *    names from step 1. The class name can be selected individually
 *    or with the keep_all_classes option.
 *
 * 3) The input detection_set is cloned and the detected_object_type
 *    from step 2 is attached.
 */

class KWIVER_ALGO_CORE_EXPORT class_probablity_filter
  : public vital::algo::detected_object_filter
{
public:
  PLUGIN_INFO( "class_probablity_filter",
               "Filters detections based on class probability.\n\n"
               "This algorithm filters out items that are less than the threshold."
               " The following steps are applied to each input detected object set.\n\n"
               "1) Select all class names with scores greater than threshold.\n\n"
               "2) Create a new detected_object_type object with all selected class"
               " names from step 1. The class name can be selected individually"
               " or with the keep_all_classes option.\n\n"
               "3) The input detection_set is cloned and the detected_object_type"
               " from step 2 is attached." )

  class_probablity_filter();
  virtual ~class_probablity_filter() = default;

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  virtual vital::detected_object_set_sptr filter( const vital::detected_object_set_sptr input_set) const;

private:
  bool m_keep_all_classes;
  std::set<std::string> m_keep_classes;
  double m_threshold;
};

}}} //End namespace

#endif
