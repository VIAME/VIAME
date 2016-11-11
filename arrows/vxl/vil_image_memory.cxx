/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief VXL image memory implementation
 */

#include "vil_image_memory.h"


namespace kwiver {
namespace arrows {
namespace vxl {

/// Create space for n bytes
void
image_memory_chunk
::set_size(unsigned long n, vil_pixel_format pixel_format)
{
  if( n != size_ )
  {
    image_data_ = vital::image_memory_sptr(new vital::image_memory(n));
    size_ = n;
  }
  pixel_format_ = pixel_format;
}


/// Convert a VXL vil_memory_chunk_sptr to a VITAL image_memory_sptr
/*
 * This conversion function typically calls the vil_image_memory constructor.
 * However, it also detects when the incoming chunk is already a wrapper around
 * vital::image_memory.  In the later case it extracts the underlying
 * vital::image_memory instead of adding another layer of wrapping.
 */
vital::image_memory_sptr vxl_to_vital(const vil_memory_chunk_sptr chunk)
{
  // prevent nested wrappers when converting back and forth.
  // if this vil_memory_chunk is already wrapping VITAL data,
  // then extract the underlying VITAL data instead of wrapping
  if( image_memory_chunk* vital_chunk =
        dynamic_cast<image_memory_chunk*>(chunk.ptr()) )
  {
    // extract the existing VITAL memory from the vil wrapper
    return vital_chunk->memory();
  }

  // create a VITAL wrapper around the vil memory chunk
  return std::make_shared<vil_image_memory>(chunk);
}


/// Convert a VITAL image_memory_sptr to a VXL vil_memory_chunk_sptr
/*
 * This conversion function typically calls the image_memory_chunk constructor.
 * However, it also detects when the incoming memory is already a wrapper around
 * vil_memory_chunk.  In the later case it extracts the underlying
 * vil_memory_chunk instead of adding another layer of wrapping.
 */
vil_memory_chunk_sptr vital_to_vxl(const vital::image_memory_sptr mem)
{
  // prevent nested wrappers when converting back and forth.
  // if this VITAL memory is already wrapping a vil chunk,
  // then extract the underlying vil data instead of wrapping
  if( vil_image_memory* vil_memory =
        dynamic_cast<vil_image_memory*>(mem.get()) )
  {
    // extract the existing vil_memory_chunk from the VITAL wrapper
    return vil_memory->memory_chunk();
  }

  // create a vil wrapper around the VITAL memory
  return new image_memory_chunk(mem);
}


} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
