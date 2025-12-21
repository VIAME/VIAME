#ifndef VIAME_OPENCV_KMEDIANS_H
#define VIAME_OPENCV_KMEDIANS_H

#include <opencv2/core.hpp>
#include <plugins/opencv/viame_opencv_export.h>

namespace viame {

VIAME_OPENCV_EXPORT int
find_closest_frame_id_to_center( const cv::Mat& data,
                                 const cv::Mat& labels,
                                 const cv::Mat& centers,
                                 int i_cluster );

VIAME_OPENCV_EXPORT int
find_furthest_frame_id_from_center( const cv::Mat& data,
                                    const cv::Mat& labels,
                                    const cv::Mat& centers,
                                    int i_cluster );

/// @brief KMedian implementation following OPENCV's KMeans interface
VIAME_OPENCV_EXPORT double
kmedians( const cv::Mat& _data,
          int K,
          cv::Mat& _bestLabels,
          cv::TermCriteria criteria,
          int attempts,
          int flags,
          cv::Mat& _centers );

} // viame


#endif // VIAME_OPENCV_KMEDIANS_H
