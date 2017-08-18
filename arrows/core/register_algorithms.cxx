/*ckwg +29
 * Copyright 2014-2017 by Kitware, Inc.
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
 * \brief Defaults plugin algorithm registration interface impl
 */

#include <arrows/core/kwiver_algo_core_plugin_export.h>
#include <vital/algo/algorithm_factory.h>

#include <arrows/core/associate_detections_to_tracks_threshold.h>
#include <arrows/core/class_probablity_filter.h>
#include <arrows/core/close_loops_bad_frames_only.h>
#include <arrows/core/close_loops_exhaustive.h>
#include <arrows/core/close_loops_keyframe.h>
#include <arrows/core/close_loops_multi_method.h>
#include <arrows/core/compute_association_matrix_from_features.h>
#include <arrows/core/compute_ref_homography_core.h>
#include <arrows/core/convert_image_bypass.h>
#include <arrows/core/estimate_canonical_transform.h>
#include <arrows/core/feature_descriptor_io.h>
#include <arrows/core/filter_features_magnitude.h>
#include <arrows/core/handle_descriptor_request_core.h>
#include <arrows/core/filter_features_scale.h>
#include <arrows/core/filter_tracks.h>
#include <arrows/core/hierarchical_bundle_adjust.h>
#include <arrows/core/initialize_cameras_landmarks.h>
#include <arrows/core/initialize_object_tracks_threshold.h>
#include <arrows/core/match_features_fundamental_matrix.h>
#include <arrows/core/match_features_homography.h>
#include <arrows/core/read_object_track_set_kw18.h>
#include <arrows/core/read_track_descriptor_set_csv.h>
#include <arrows/core/track_features_core.h>
#include <arrows/core/triangulate_landmarks.h>
#include <arrows/core/video_input_filter.h>
#include <arrows/core/video_input_image_list.h>
#include <arrows/core/video_input_pos.h>
#include <arrows/core/video_input_split.h>
#include <arrows/core/write_object_track_set_kw18.h>
#include <arrows/core/write_track_descriptor_set_csv.h>

#include <arrows/core/detected_object_set_input_kw18.h>
#include <arrows/core/detected_object_set_output_kw18.h>
#include <arrows/core/detected_object_set_input_csv.h>
#include <arrows/core/detected_object_set_output_csv.h>
#include <arrows/core/dynamic_config_none.h>


namespace kwiver {
namespace arrows {
namespace core {

extern "C"
KWIVER_ALGO_CORE_PLUGIN_EXPORT
void
register_factories( kwiver::vital::plugin_loader& vpm )
{
  static auto const module_name = std::string( "arrows.core" );
  if (vpm.is_module_loaded( module_name ) )
  {
    return;
  }

  // add factory                  implementation-name       type-to-create
  auto fact = vpm.ADD_ALGORITHM( "bad_frames_only", kwiver::arrows::core::close_loops_bad_frames_only );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Attempts short-term loop closure based on percentage of feature."
                    "points tracked.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "exhaustive", kwiver::arrows::core::close_loops_exhaustive );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                    "Exhaustive matching of all frame pairs, or all frames within a moving window." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "keyframe", kwiver::arrows::core::close_loops_keyframe );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Establishes keyframes matches to all keyframes.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "multi_method", kwiver::arrows::core::close_loops_multi_method );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Iteratively run multiple loop closure algorithms." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::compute_ref_homography_core );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Default online sequential-frame reference homography estimator." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "bypass", kwiver::arrows::core::convert_image_bypass );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Performs no conversion and returns the given image container." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core_pca", kwiver::arrows::core::estimate_canonical_transform );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Uses PCA to estimate a canonical similarity transform that aligns the best fit plane to Z=0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::feature_descriptor_io );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Read and write features and descriptor to binary files using Cereal serialization." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "magnitude", kwiver::arrows::core::filter_features_magnitude );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Filter features using a threshold on the magnitude of the detector response function." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "scale", kwiver::arrows::core::filter_features_scale );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Filter features using a threshold on the scale of the detected features." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::filter_tracks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Filter tracks by track length or matrix matrix importance." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "hierarchical", kwiver::arrows::core::hierarchical_bundle_adjust );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Run a bundle adjustment algorithm in a temporally hierarchical fashion (useful for video)" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::initialize_cameras_landmarks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Run SfM to iteratively estimate new cameras and landmarks using feature tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "fundamental_matrix_guided", kwiver::arrows::core::match_features_fundamental_matrix );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use an estimated fundamental matrix as a geometric filter to remove outlier matches." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "homography_guided", kwiver::arrows::core::match_features_homography );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Use an estimated homography as a geometric filter to remove outlier matches." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::track_features_core );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Track features from frame to frame using feature detection, matching, and loop closure." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::triangulate_landmarks );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Triangulate landmarks from tracks and cameras using a simple least squares solver." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "core", kwiver::arrows::core::handle_descriptor_request_core );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Formulate descriptors for later queries." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "filter", kwiver::arrows::core::video_input_filter );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "A video input that calls another video input and filters the output on "
                       "frame range and other parameters." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "image_list", kwiver::arrows::core::video_input_image_list );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Read a list of images from a list of file names and presents them in the same way "
                       "as reading a video.  The actual algorithm to read an image is specified in the "
                       "\"image_reader\" config block.  Read an image list as a video stream." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "pos", kwiver::arrows::core::video_input_pos );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Read video metadata in AFRL POS format. "
                       "The algorithm takes configuration for a directory full of images "
                       "and an associated directory name for the metadata files. These "
                       "metadata files have the same base name as the image files. "
                       "Each metadata file is associated with the image file." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "split", kwiver::arrows::core::video_input_split );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Coordinate two video readers. One reader supplies the image/data stream. "
                       "The other reader supplies the metadata stream." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "none", kwiver::arrows::core::dynamic_config_none );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Null implementation of dynamic_configuration.\n\n"
                       "This algorithm always returns an empty configuration block.")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "kw18", kwiver::arrows::core::detected_object_set_input_kw18 );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Detected object set reader using kw18 format.\n\n"
                       "  - Column(s) 1: Track-id\n"
                       "  - Column(s) 2: Track-length (# of detections)\n"
                       "  - Column(s) 3: Frame-number (-1 if not available)\n"
                       "  - Column(s) 4-5: Tracking-plane-loc(x,y) (Could be same as World-loc)\n"
                       "  - Column(s) 6-7: Velocity(x,y)\n"
                       "  - Column(s) 8-9: Image-loc(x,y)\n"
                       "  - Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left & bottom-right vertices)\n"
                       "  - Column(s) 14: Area\n"
                       "  - Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when available)\n"
                       "  - Column(s) 18: Timesetamp(-1 if not available)\n"
                       "  - Column(s) 19: Track-confidence(-1_when_not_available)\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "kw18", kwiver::arrows::core::detected_object_set_output_kw18 );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Detected object set writer using kw18 format.\n\n"
                       "  - Column(s) 1: Track-id\n"
                       "  - Column(s) 2: Track-length (# of detections)\n"
                       "  - Column(s) 3: Frame-number (-1 if not available)\n"
                       "  - Column(s) 4-5: Tracking-plane-loc(x,y) (Could be same as World-loc)\n"
                       "  - Column(s) 6-7: Velocity(x,y)\n"
                       "  - Column(s) 8-9: Image-loc(x,y)\n"
                       "  - Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left & bottom-right vertices)\n"
                       "  - Column(s) 14: Area\n"
                       "  - Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when available)\n"
                       "  - Column(s) 18: Timesetamp(-1 if not available)\n"
                       "  - Column(s) 19: Track-confidence(-1_when_not_available)\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "csv", kwiver::arrows::core::detected_object_set_input_csv );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Detected object set reader using CSV format.\n\n"
                       " - 1: frame number\n"
                       " - 2: file name\n"
                       " - 3: TL-x\n"
                       " - 4: TL-y\n"
                       " - 5: BR-x\n"
                       " - 6: BR-y\n"
                       " - 7: confidence\n"
                       " - 8,9  : class-name,  score  (this pair may be omitted or may repeat any number of times)\n")

    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;


  fact = vpm.ADD_ALGORITHM( "csv", kwiver::arrows::core::detected_object_set_output_csv );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Detected object set reader using CSV format.\n\n"
                       " - 1: frame number\n"
                       " - 2: file name\n"
                       " - 3: TL-x\n"
                       " - 4: TL-y\n"
                       " - 5: BR-x\n"
                       " - 6: BR-y\n"
                       " - 7: confidence\n"
                       " - 8,9  : class-name,  score  (this pair may be omitted or may repeat any number of times)\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "csv", kwiver::arrows::core::read_track_descriptor_set_csv );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Track descriptor set csv reader\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "csv", kwiver::arrows::core::write_track_descriptor_set_csv );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Track descriptor set csv writer\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "kw18", kwiver::arrows::core::read_object_track_set_kw18 );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Object track set kw18 reader\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "kw18", kwiver::arrows::core::write_object_track_set_kw18 );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Object track set kw18 writer\n")
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "class_probablity_filter", kwiver::arrows::core::class_probablity_filter );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Filters detections based on class probability.\n\n"
                       "This algorithm filters out items that are less than the threshold. "
                       "The following steps are applied to each input detected object set.\n\n"
                       "1) Select all class names with scores greater than threshold.\n\n"
                       "2) Create a new detected_object_type object with all selected class "
                       "names from step 1. The class name can be selected individually "
                       "or with the keep_all_classes option.\n\n"
                       "3) The input detection_set is cloned and the detected_object_type "
                       "from step 2 is attached." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "threshold", kwiver::arrows::core::associate_detections_to_tracks_threshold );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Associate detections to tracks via simple thresholding on the input matrix." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "from_features", kwiver::arrows::core::compute_association_matrix_from_features );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Populate association matrix in tracking from detector features." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  fact = vpm.ADD_ALGORITHM( "threshold", kwiver::arrows::core::initialize_object_tracks_threshold );
  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION,
                       "Perform thresholding on detection confidence values to create tracks." )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, module_name )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_VERSION, "1.0" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_ORGANIZATION, "Kitware Inc." )
    ;

  vpm.mark_module_as_loaded( module_name );
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
