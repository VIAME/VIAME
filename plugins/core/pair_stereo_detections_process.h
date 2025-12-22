/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
