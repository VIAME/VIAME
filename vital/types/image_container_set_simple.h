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
 * \brief Interface for simple implementation of image_container_set
 */

#ifndef VITAL_IMAGE_CONTAINER_SET_SIMPLE_H_
#define VITAL_IMAGE_CONTAINER_SET_SIMPLE_H_

#include <vital/types/image_container_set.h>


namespace kwiver {
namespace vital {


/// A concrete image container set that simply wraps a vector of images.
class VITAL_EXPORT simple_image_container_set
  : public image_container_set
{
public:
  /// Default Constructor
  simple_image_container_set();

  /// Constructor from a vector of images
  explicit simple_image_container_set( std::vector< image_container_sptr > const& images );

  /// Return the number of items
  size_t size() const override;
  bool empty() const override;
  image_container_sptr at( size_t index ) override;
  image_container_sptr const at( size_t index ) const override;

protected:
  using vec_t = std::vector< image_container_sptr >;

  /// The vector of images
  vec_t data_;

  /// Implement next function for non-const iterator.
  iterator::next_value_func_t get_iter_next_func();

  /// Implement next function for const iterator.
  const_iterator::next_value_func_t get_const_iter_next_func() const;
};


} } // end namespaces


#endif // VITAL_IMAGE_CONTAINER_SET_SIMPLE_H_
