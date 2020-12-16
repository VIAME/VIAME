// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV mat_image_memory interface
 */

#ifndef KWIVER_ARROWS_OCV_MAT_IMAGE_MEMORY_H_
#define KWIVER_ARROWS_OCV_MAT_IMAGE_MEMORY_H_

#include <vital/vital_config.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/types/image.h>

#include <opencv2/core/core.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

/// An image memory class that shares memory with OpenCV using reference counting
class KWIVER_ALGO_OCV_EXPORT mat_image_memory
  : public vital::image_memory
{
public:
  /// Constructor - allocates n bytes
  mat_image_memory(const cv::Mat& m);

  /// Destructor
  virtual ~mat_image_memory();

  /// Return a pointer to the allocated memory
  virtual void* data() { return this->mat_data_; }

  /// Return the OpenCV reference counter
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  int* get_ref_counter() const { return this->mat_refcount_; }
#else
  cv::UMatData* get_umatdata() const { return this->u_; }
#endif

protected:
  /// The cv::Mat data
  unsigned char* mat_data_;

  /// The ref count shared with cv::Mat
#if KWIVER_OPENCV_VERSION_MAJOR < 3
  int* mat_refcount_;
#else
  cv::UMatData *u_;
#endif
};

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
