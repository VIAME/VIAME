/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_DARKNET_CUSTOM_RESIZE_H
#define VIAME_DARKNET_CUSTOM_RESIZE_H

#include <plugins/darknet/viame_darknet_export.h>

#include <vital/algo/image_object_detector.h>

#include <opencv2/core/core.hpp>

namespace viame {

double
scale_image_maintaining_ar( const cv::Mat& src, cv::Mat& dst,
                            int width, int height );

double
format_image( const cv::Mat& src, cv::Mat& dst, std::string option,
              double scale_factor, int width, int height );

} // end namespace

#endif /* VIAME_DARKNET_CUSTOM_RESIZE_H */
