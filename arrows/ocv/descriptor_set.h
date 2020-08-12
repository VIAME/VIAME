/*ckwg +29
 * Copyright 2013-2016, 2019 by Kitware, Inc.
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
