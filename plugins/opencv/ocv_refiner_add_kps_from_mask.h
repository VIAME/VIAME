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
 * \brief Algorithm to add keypoints to detections from mask
 */

#ifndef VIAME_OCV_REFINER_ADD_KPS_FROM_MASK_H
#define VIAME_OCV_REFINER_ADD_KPS_FROM_MASK_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/refine_detections.h>

namespace viame
{

// -----------------------------------------------------------------------------
/**
 * @brief Algorithm that adds head/tail keypoints to detections based on their
 *        mask or bounding box.
 *
 * This algorithm takes a detection set as input, computes keypoints using one
 * of several methods (oriented bounding box, PCA, farthest points, hull extremes,
 * or skeleton), and adds head/tail keypoints. The head keypoint is placed at
 * the end with the larger x coordinate.
 */
class VIAME_OPENCV_EXPORT ocv_refiner_add_kps_from_mask
  : public kwiver::vital::algorithm_impl<
      ocv_refiner_add_kps_from_mask,
      kwiver::vital::algo::refine_detections >
{
public:
  PLUGIN_INFO( "add_keypoints_from_mask",
    "Adds head and tail keypoints to detections based on their "
    "mask or bounding box using configurable methods." )

  ocv_refiner_add_kps_from_mask();
  virtual ~ocv_refiner_add_kps_from_mask();

  virtual kwiver::vital::config_block_sptr get_configuration() const override;
  virtual void set_configuration( kwiver::vital::config_block_sptr config ) override;
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const override;

  virtual kwiver::vital::detected_object_set_sptr
  refine( kwiver::vital::image_container_sptr image_data,
          kwiver::vital::detected_object_set_sptr detections ) const override;

private:
  class priv;
  const std::unique_ptr<priv> d;

}; // end class ocv_refiner_add_kps_from_mask

} // end namespace viame

#endif // VIAME_OCV_REFINER_ADD_KPS_FROM_MASK_H
