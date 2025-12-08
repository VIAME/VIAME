/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
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
 * \brief Process to add keypoints to detections from oriented bounding box
 */

#ifndef VIAME_OCV_ADD_KEYPOINTS_FROM_POLY_H
#define VIAME_OCV_ADD_KEYPOINTS_FROM_POLY_H

#include <sprokit/pipeline/process.h>

#include <plugins/opencv/viame_opencv_export.h>

#include <memory>

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Process that adds head/tail keypoints to detections based on their
 *        mask or bounding box.
 *
 * This process takes a detection set as input, computes an oriented bounding
 * box from each detection's mask (or uses the axis-aligned bounding box if
 * no mask is present), and adds head/tail keypoints at the midpoints of the
 * short edges. The head keypoint is placed at the end with the larger x
 * coordinate.
 */
class VIAME_OPENCV_NO_EXPORT ocv_add_keypoints_from_poly
  : public sprokit::process
{
public:
  // -- CONSTRUCTORS --
  ocv_add_keypoints_from_poly( kwiver::vital::config_block_sptr const& config );
  virtual ~ocv_add_keypoints_from_poly();

protected:
  void _configure() override;
  void _step() override;

private:
  void make_ports();
  void make_config();

  class priv;
  const std::unique_ptr<priv> d;

}; // end class ocv_add_keypoints_from_poly

} // end namespace viame

#endif // VIAME_OCV_ADD_KEYPOINTS_FROM_POLY_H
