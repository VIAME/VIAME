/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_APPLY_COLOR_CORRECTION_H
#define VIAME_OPENCV_APPLY_COLOR_CORRECTION_H

#include "viame_opencv_export.h"

#include <vital/algo/image_filter.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include <opencv2/core/core.hpp>

namespace viame {

class VIAME_OPENCV_EXPORT apply_color_correction
  : public kwiver::vital::algo::image_filter
{
public:
  PLUGGABLE_IMPL(
    apply_color_correction,
    "Color correction algorithms: gamma, underwater compensation, gray world white balance",
    PARAM_DEFAULT( apply_gamma, bool,
      "Enable gamma correction", false ),
    PARAM_DEFAULT( gamma, double,
      "Gamma value. Less than 1.0 brightens image, greater than 1.0 darkens", 1.0 ),
    PARAM_DEFAULT( gamma_auto, bool,
      "Automatically estimate optimal gamma from image histogram", false ),
    PARAM_DEFAULT( apply_gray_world, bool,
      "Enable gray world white balance algorithm", false ),
    PARAM_DEFAULT( gray_world_sat_threshold, double,
      "Exclude pixels above this threshold (0-1) from white balance calculation", 0.95 ),
    PARAM_DEFAULT( apply_underwater, bool,
      "Enable underwater color correction", false ),
    PARAM_DEFAULT( underwater_method, std::string,
      "Underwater correction method: 'simple' or 'fusion'", "simple" ),
    PARAM_DEFAULT( depth_map_path, std::string,
      "Path to precomputed depth map image (optional)", "" ),
    PARAM_DEFAULT( use_auto_depth, bool,
      "Automatically estimate relative depth when no depth map provided", true ),
    PARAM_DEFAULT( water_type, std::string,
      "Water type preset: 'oceanic', 'coastal', or 'turbid'", "oceanic" ),
    PARAM_DEFAULT( red_attenuation, double,
      "Red channel attenuation coefficient (0-1)", 0.5 ),
    PARAM_DEFAULT( green_attenuation, double,
      "Green channel attenuation coefficient (0-1)", 0.3 ),
    PARAM_DEFAULT( blue_attenuation, double,
      "Blue channel attenuation coefficient (0-1)", 0.1 ),
    PARAM_DEFAULT( backscatter_removal, bool,
      "Apply backscatter (haze) removal in underwater correction", true )
  )

  virtual ~apply_color_correction() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main filtering method
  virtual kwiver::vital::image_container_sptr filter(
    kwiver::vital::image_container_sptr image_data );

private:
  // Non-config member variables
  cv::Mat m_depth_map;

  // Helper methods
  void apply_gamma_correction( cv::Mat& image );
  double estimate_auto_gamma( const cv::Mat& image );
  void apply_gray_world_balance( cv::Mat& image );
  void apply_underwater_simple( cv::Mat& image );
  void apply_underwater_fusion( cv::Mat& image );
  cv::Mat estimate_relative_depth( const cv::Mat& image );
  void apply_backscatter_removal( cv::Mat& image );
  void load_depth_map();
  void set_water_type_presets();
};

} // end namespace

#endif /* VIAME_OPENCV_APPLY_COLOR_CORRECTION_H */
