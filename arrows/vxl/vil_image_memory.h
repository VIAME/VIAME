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
 * \brief VXL image memory interface
 */

#ifndef KWIVER_ARROWS_VXL_VIL_IMAGE_MEMORY_H_
#define KWIVER_ARROWS_VXL_VIL_IMAGE_MEMORY_H_


#include <vital/vital_config.h>
#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/types/image.h>

#include <vil/vil_memory_chunk.h>


namespace kwiver {
namespace arrows {
namespace vxl {

/// An image memory class that shares memory with VXL using vil_memory_chunk
class KWIVER_ALGO_VXL_EXPORT vil_image_memory
  : public vital::image_memory
{
public:
  /// Constructor - allocates n bytes
  vil_image_memory(vil_memory_chunk_sptr chunk)
  : vil_data_(chunk)
  {
    size_ = chunk->size();
  }

  /// Return a pointer to the allocated memory
  virtual void* data() { return vil_data_->data(); }

  /// Return the underlying vil_memory_chunk
  vil_memory_chunk_sptr memory_chunk() const { return vil_data_; }

protected:
  /// The vil image data
  vil_memory_chunk_sptr vil_data_;

};


/// An image memory class that shares memory with vital using image_memory
class KWIVER_ALGO_VXL_EXPORT image_memory_chunk
 : public vil_memory_chunk
{
public:
  /// Constructor - from image memory
  image_memory_chunk(vital::image_memory_sptr mem)
  : image_data_(mem)
  {
    size_ = image_data_->size();
    pixel_format_ = VIL_PIXEL_FORMAT_BYTE;
  }

  /// Pointer to the first element of data
  virtual void* data() { return image_data_->data(); }

  /// Pointer to the first element of data
  virtual void* const_data() const { return image_data_->data(); }

  /// Create space for n bytes
  virtual void set_size(unsigned long n, vil_pixel_format pixel_format);

  /// Access the underlying image memory
  vital::image_memory_sptr memory() const { return image_data_; }
protected:

  /// The image data
  vital::image_memory_sptr image_data_;
};


/// Convert a VXL vil_memory_chunk_sptr to a VITAL image_memory_sptr
/*
 * This conversion function typically calls the vil_image_memory constructor.
 * However, it also detects when the incoming chunk is already a wrapper around
 * vital::image_memory.  In the later case it extracts the underlying
 * vital::image_memory instead of adding another layer of wrapping.
 */
vital::image_memory_sptr vxl_to_vital(const vil_memory_chunk_sptr chunk);


/// Convert a VITAL image_memory_sptr to a VXL vil_memory_chunk_sptr
/*
 * This conversion function typically calls the image_memory_chunk constructor.
 * However, it also detects when the incoming memory is already a wrapper around
 * vil_memory_chunk.  In the later case it extracts the underlying
 * vil_memory_chunk instead of adding another layer of wrapping.
 */
vil_memory_chunk_sptr vital_to_vxl(const vital::image_memory_sptr mem);


} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
