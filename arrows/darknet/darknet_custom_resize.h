// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_DARKENT_CUSTOM_RESIZE
#define KWIVER_ARROWS_DARKENT_CUSTOM_RESIZE

#include <arrows/darknet/kwiver_algo_darknet_export.h>

#include <opencv2/core/core.hpp>

#include <vital/algo/image_object_detector.h>

namespace kwiver {
namespace arrows {
namespace darknet {

double
scale_image_maintaining_ar( const cv::Mat& src, cv::Mat& dst,
                            int width, int height );

double
format_image( const cv::Mat& src, cv::Mat& dst, std::string option,
              double scale_factor, int width, int height );

} } }

#endif /* KWIVER_ARROWS_DARKENT_DETECTOR */
