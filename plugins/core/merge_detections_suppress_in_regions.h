// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VIAME_CORE_MERGE_DETECTIONS_SUPPRESS_IN_REGIONS_H
#define VIAME_CORE_MERGE_DETECTIONS_SUPPRESS_IN_REGIONS_H

#include "viame_core_export.h"

#include <vital/algo/merge_detections.h>

namespace viame {

// -----------------------------------------------------------------------------
/**
 * \class merge_detections_suppress_in_regions
 *
 * \brief Prunes detections overlapping with regions identified by class string
 */
class VIAME_CORE_EXPORT merge_detections_suppress_in_regions
  : public kwiver::vital::algorithm_impl< merge_detections_suppress_in_regions,
    kwiver::vital::algo::merge_detections >
{

public:
  PLUGIN_INFO( "suppress_in_regions",
    "Suppresses detections within regions indicated by a certain fixed category "
    "of detections. Can either remove the detections or reduce their probability." )

  merge_detections_suppress_in_regions();
  virtual ~merge_detections_suppress_in_regions();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
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

  /// private implementation class
  class priv;
  const std::unique_ptr< priv > d;

}; // end class merge_detections_suppress_in_regions

} // end namespace viame

#endif // VIAME_CORE_MERGE_DETECTIONS_SUPPRESS_IN_REGIONS_H
