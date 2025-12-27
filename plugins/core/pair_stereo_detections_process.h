/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo detection pairing process
 *
 * This process matches detections across stereo camera views using one of several
 * methods: IOU (bounding box overlap), calibration-based (stereo geometry), or
 * feature matching (descriptor-based with homography filtering).
 *
 * Input can be either detected_object_set or object_track_set ports. When using
 * track inputs, detections are extracted from each track's state for the current
 * frame. The process auto-detects which input type is connected.
 *
 * When compute_head_tail_points is enabled and images are provided, the process
 * will detect and match features between paired detections, apply RANSAC-based
 * outlier rejection, and add the two furthest apart inlier points as head/tail
 * keypoints to the detection objects for use in downstream stereo measurement.
 */

#ifndef VIAME_CORE_PAIR_STEREO_DETECTIONS_PROCESS_H
#define VIAME_CORE_PAIR_STEREO_DETECTIONS_PROCESS_H

#include <sprokit/pipeline/process.h>

#include <plugins/core/viame_processes_core_export.h>

#include <memory>

namespace viame
{

namespace core
{

// -----------------------------------------------------------------------------
/**
 * @brief Stereo detection pairing process
 *
 * Matches detections from two stereo camera views and outputs track sets with
 * aligned track IDs for matched detection pairs. Unmatched detections are
 * assigned unique track IDs.
 *
 * Inputs: Either detected_object_set1/2 or object_track_set1/2 ports can be
 * connected. The process auto-detects which type is connected. When using
 * track inputs, detections are extracted from each track's current frame state.
 * For feature-based matching or head/tail computation, image1/2 ports must also
 * be connected.
 *
 * Matching methods:
 * - iou: Bounding box IOU (Intersection over Union)
 * - calibration: Stereo geometry and reprojection error
 * - feature_matching: Feature detection and descriptor matching
 *
 * Optional features:
 * - compute_head_tail_points: When enabled, computes head/tail keypoints from
 *   the two furthest apart inlier feature matches after RANSAC filtering.
 *   These keypoints are added to both paired detections for downstream stereo
 *   measurement algorithms.
 */
class VIAME_PROCESSES_CORE_NO_EXPORT pair_stereo_detections_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  pair_stereo_detections_process( kwiver::vital::config_block_sptr const& config );
  virtual ~pair_stereo_detections_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class pair_stereo_detections_process

} // end namespace core
} // end namespace viame

#endif // VIAME_CORE_PAIR_STEREO_DETECTIONS_PROCESS_H
