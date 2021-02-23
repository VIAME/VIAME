/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief OpenCV algorithm registration implementation
 */

#include <arrows/ocv/kwiver_algo_ocv_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_NONFREE
#include <opencv2/nonfree/nonfree.hpp>
#endif

#include <arrows/ocv/analyze_tracks.h>
#include <arrows/ocv/convert_color_space.h>
#include <arrows/ocv/detect_features_AGAST.h>
#include <arrows/ocv/detect_features_FAST.h>
#include <arrows/ocv/detect_features_GFTT.h>
#include <arrows/ocv/detect_features_MSD.h>
#include <arrows/ocv/detect_features_MSER.h>
#include <arrows/ocv/detect_features_simple_blob.h>
#include <arrows/ocv/detect_features_STAR.h>
#include <arrows/ocv/draw_detected_object_set.h>
#include <arrows/ocv/draw_tracks.h>
#include <arrows/ocv/estimate_fundamental_matrix.h>
#include <arrows/ocv/estimate_homography.h>
//#include <arrows/ocv/estimate_pnp.h>
#include <arrows/ocv/extract_descriptors_BRIEF.h>
#include <arrows/ocv/extract_descriptors_DAISY.h>
#include <arrows/ocv/extract_descriptors_FREAK.h>
#include <arrows/ocv/extract_descriptors_LATCH.h>
#include <arrows/ocv/extract_descriptors_LUCID.h>
#include <arrows/ocv/feature_detect_extract_BRISK.h>
#include <arrows/ocv/feature_detect_extract_ORB.h>
#include <arrows/ocv/feature_detect_extract_SIFT.h>
#include <arrows/ocv/feature_detect_extract_SURF.h>
#include <arrows/ocv/image_io.h>
#include <arrows/ocv/match_features_bruteforce.h>
#include <arrows/ocv/match_features_flannbased.h>
#include <arrows/ocv/merge_images.h>
#include <arrows/ocv/hough_circle_detector.h>
#include <arrows/ocv/refine_detections_grabcut.h>
#include <arrows/ocv/refine_detections_watershed.h>
#include <arrows/ocv/refine_detections_write_to_disk.h>
#include <arrows/ocv/split_image_channels.h>
#include <arrows/ocv/split_image_horizontally.h>
#include <arrows/ocv/track_features_klt.h>
#include <arrows/ocv/detect_motion_3frame_differencing.h>
#include <arrows/ocv/detect_motion_mog2.h>
#include <arrows/ocv/detect_heat_map.h>
#include <arrows/ocv/windowed_detector.h>
#include <arrows/ocv/windowed_trainer.h>

namespace kwiver {
namespace arrows {
namespace ocv {

extern "C"
KWIVER_ALGO_OCV_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  kwiver::vital::algorithm_registrar reg( vpm, "arrows.ocv" );

  if (reg.is_module_loaded())
  {
    return;
  }

#if defined(HAVE_OPENCV_NONFREE)
  cv::initModule_nonfree();
#endif

  reg.register_algorithm< analyze_tracks >();
  reg.register_algorithm< convert_color_space >();
  reg.register_algorithm< draw_tracks >();
  reg.register_algorithm< estimate_fundamental_matrix >();
  reg.register_algorithm< estimate_homography >();
  reg.register_algorithm< image_io >();
  reg.register_algorithm< draw_detected_object_set >();

  reg.register_algorithm< detect_features_BRISK >();
  reg.register_algorithm< detect_features_FAST >();
  reg.register_algorithm< detect_features_GFTT >();
  reg.register_algorithm< detect_features_MSER >();
  reg.register_algorithm< detect_features_ORB >();
  reg.register_algorithm< detect_features_simple_blob >();

  reg.register_algorithm< extract_descriptors_BRISK >();
  reg.register_algorithm< extract_descriptors_ORB >();

  reg.register_algorithm< match_features_bruteforce >();
  reg.register_algorithm< match_features_flannbased >();

  reg.register_algorithm< hough_circle_detector >();
  reg.register_algorithm< detect_motion_3frame_differencing >();
  reg.register_algorithm< detect_motion_mog2 >();

  reg.register_algorithm< windowed_detector >();
  reg.register_algorithm< windowed_trainer >();

  // Conditional algorithms
  // Source ``KWIVER_OCV_HAS_*`` symbol definitions can be found in the header
  //  files of the algorithms referred to.
#ifdef KWIVER_OCV_HAS_AGAST
  reg.register_algorithm< detect_features_AGAST >();
#endif

#ifdef KWIVER_OCV_HAS_BRIEF
  reg.register_algorithm< extract_descriptors_BRIEF >();
#endif

#ifdef KWIVER_OCV_HAS_DAISY
  reg.register_algorithm< extract_descriptors_DAISY >();
#endif

#ifdef KWIVER_OCV_HAS_FREAK
    reg.register_algorithm< extract_descriptors_FREAK >();
#endif

#ifdef KWIVER_OCV_HAS_LATCH
   reg.register_algorithm< extract_descriptors_LATCH >();
#endif

#ifdef KWIVER_OCV_HAS_LUCID
  reg.register_algorithm< extract_descriptors_LUCID >();
#endif

#ifdef KWIVER_OCV_HAS_MSD
  reg.register_algorithm< detect_features_MSD >();
#endif

#ifdef KWIVER_OCV_HAS_SIFT
  reg.register_algorithm< detect_features_SIFT >();
  reg.register_algorithm< extract_descriptors_SIFT >();
#endif

#ifdef KWIVER_OCV_HAS_STAR
  reg.register_algorithm< detect_features_STAR >();
#endif

#ifdef KWIVER_OCV_HAS_SURF
  reg.register_algorithm< detect_features_SURF >();
  reg.register_algorithm< extract_descriptors_SURF >();
#endif

  reg.register_algorithm< detect_heat_map >();

  reg.register_algorithm< refine_detections_grabcut >();
  reg.register_algorithm< refine_detections_watershed >();
  reg.register_algorithm< refine_detections_write_to_disk >();
  reg.register_algorithm< split_image_channels >();
  reg.register_algorithm< split_image_horizontally >();
  reg.register_algorithm< merge_images >();
  reg.register_algorithm< track_features_klt >();
  //reg.register_algorithm< estimate_pnp >();

  reg.mark_module_as_loaded();
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
