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
 * This process matches detections across stereo camera views without requiring
 * OpenCV or calibration data. It uses simple criteria like bounding box overlap
 * and class label matching to find corresponding detections.
 */

#ifndef VIAME_STEREO_DETECTION_PAIRING_PROCESS_H
#define VIAME_STEREO_DETECTION_PAIRING_PROCESS_H

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
 * Matching is based on:
 * - Class label matching (optional)
 * - Bounding box IOU (Intersection over Union)
 * - Greedy or optimal assignment
 */
class VIAME_PROCESSES_CORE_NO_EXPORT stereo_detection_pairing_process
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  stereo_detection_pairing_process( kwiver::vital::config_block_sptr const& config );
  virtual ~stereo_detection_pairing_process();

protected:
  virtual void _configure();
  virtual void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class stereo_detection_pairing_process

} // end namespace core
} // end namespace viame

#endif // VIAME_STEREO_DETECTION_PAIRING_PROCESS_H
