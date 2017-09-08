/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
#include <arrows/ocv/hough_circle_detector.h>
#include <arrows/ocv/refine_detections_write_to_disk.h>
#include <arrows/ocv/split_image.h>

namespace kwiver {
namespace arrows {
namespace ocv {

extern "C"
KWIVER_ALGO_OCV_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.ocv" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

#if defined(HAVE_OPENCV_NONFREE)
  cv::initModule_nonfree();
#endif

  // add factory               implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::analyze_tracks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Use OpenCV to analyze statistics of feature tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::draw_tracks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Use OpenCV to draw tracked features on the images." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::estimate_fundamental_matrix );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Use OpenCV to estimate a fundimental matrix from feature matches." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::estimate_homography );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Use OpenCV to estimate a homography from feature matches." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::image_io );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Read and write image using OpenCV." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::draw_detected_object_set );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Draw bounding box around detected objects on supplied image." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  // OCV Algorithm based class wrappers
  fact = vpm.ADD_ALGORITHM( "ocv_BRISK", kwiver::arrows::ocv::detect_features_BRISK );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the BRISK algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_FAST", kwiver::arrows::ocv::detect_features_FAST );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the FAST algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_GFTT", kwiver::arrows::ocv::detect_features_GFTT );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the GFTT algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_MSER", kwiver::arrows::ocv::detect_features_MSER );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the MSER algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_ORB", kwiver::arrows::ocv::detect_features_ORB );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the ORB algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_simple_blob", kwiver::arrows::ocv::detect_features_simple_blob );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the simple_blob algorithm." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_BRISK", kwiver::arrows::ocv::extract_descriptors_BRISK );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the BRISK algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_ORB", kwiver::arrows::ocv::extract_descriptors_ORB );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the ORB algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_brute_force", kwiver::arrows::ocv::match_features_bruteforce );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "OpenCV feature matcher using brute force matching (exhaustive search)." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_flann_based", kwiver::arrows::ocv::match_features_flannbased );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature matcher using FLANN (Approximate Nearest Neighbors).")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "hough_circle", kwiver::arrows::ocv::hough_circle_detector );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  // Conditional algorithms
  // Source ``KWIVER_OCV_HAS_*`` symbol definitions can be found in the header
  //  files of the algorithms referred to.
#ifdef KWIVER_OCV_HAS_AGAST
  fact = vpm.ADD_ALGORITHM( "ocv_AGAST", kwiver::arrows::ocv::detect_features_AGAST );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the AGAST algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_BRIEF
  fact = vpm.ADD_ALGORITHM( "ocv_BRIEF", kwiver::arrows::ocv::extract_descriptors_BRIEF );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the BRIEF algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_DAISY
  fact = vpm.ADD_ALGORITHM( "ocv_DAISY", kwiver::arrows::ocv::extract_descriptors_DAISY );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the DAISY algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_FREAK
  fact = vpm.ADD_ALGORITHM( "ocv_FREAK", kwiver::arrows::ocv::extract_descriptors_FREAK );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the FREAK algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_LATCH
  fact = vpm.ADD_ALGORITHM( "ocv_LATCH", kwiver::arrows::ocv::extract_descriptors_LATCH );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the LATCH algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_LUCID
  fact = vpm.ADD_ALGORITHM( "ocv_LUCID", kwiver::arrows::ocv::extract_descriptors_LUCID);
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the LUCID algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_MSD
  fact = vpm.ADD_ALGORITHM( "ocv_MSD", kwiver::arrows::ocv::detect_features_MSD );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the MSD algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_SIFT
  fact = vpm.ADD_ALGORITHM( "ocv_SIFT", kwiver::arrows::ocv::detect_features_SIFT );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the SIFT algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "ocv_SIFT", kwiver::arrows::ocv::extract_descriptors_SIFT );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the SIFT algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_STAR
  fact = vpm.ADD_ALGORITHM( "ocv_STAR", kwiver::arrows::ocv::detect_features_STAR );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the STAR algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

#ifdef KWIVER_OCV_HAS_SURF
  fact = vpm.ADD_ALGORITHM( "ocv_SURF", kwiver::arrows::ocv::detect_features_SURF );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature detection via the SURF algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "ocv_SURF", kwiver::arrows::ocv::extract_descriptors_SURF );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "OpenCV feature-point descriptor extraction via the SURF algorithm" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;
#endif

  fact = vpm.ADD_ALGORITHM( "ocv_write", kwiver::arrows::ocv::refine_detections_write_to_disk );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Debugging process for writing out detections" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "ocv", kwiver::arrows::ocv::split_image );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Split an image  into multiple smaller images using opencv functions" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
