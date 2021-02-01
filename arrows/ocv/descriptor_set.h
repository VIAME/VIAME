// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV descriptor_set interface
 */

#ifndef KWIVER_ARROWS_OCV_DESCRIPTOR_SET_H_
#define KWIVER_ARROWS_OCV_DESCRIPTOR_SET_H_

#include <vital/vital_config.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/types/descriptor_set.h>

#include <opencv2/features2d/features2d.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

/// A concrete descriptor set that wraps OpenCV descriptors.
class KWIVER_ALGO_OCV_EXPORT descriptor_set
  : public vital::descriptor_set
{
public:
  /// Default Constructor
  descriptor_set() {}

  /// Constructor from an OpenCV descriptor matrix
  explicit descriptor_set(const cv::Mat& descriptor_matrix)
  : data_(descriptor_matrix) {}

  /// Return the number of descriptor in the set
  virtual size_t size() const override { return data_.rows; }
  virtual bool empty() const override { return size() == 0; }

  /// Return a vector of descriptor shared pointers
  virtual std::vector<vital::descriptor_sptr> descriptors() const;

  /// Return the native OpenCV descriptors as a matrix
  const cv::Mat& ocv_desc_matrix() const { return data_; }

  vital::descriptor_sptr at( size_t index ) override;
  vital::descriptor_sptr const at( size_t index ) const override;

protected:
  iterator::next_value_func_t get_iter_next_func() override;
  const_iterator::next_value_func_t get_const_iter_next_func() const override;

  /// The OpenCV matrix of featrues
  cv::Mat data_;
};

/// Convert any descriptor set to an OpenCV cv::Mat
/**
 * \param desc_set descriptors to convert to cv::mat
 */
KWIVER_ALGO_OCV_EXPORT cv::Mat
descriptors_to_ocv_matrix(const vital::descriptor_set& desc_set);

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif
