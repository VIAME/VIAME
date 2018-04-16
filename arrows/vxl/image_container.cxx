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
 * \brief VXL image container implementation
 */

#include "image_container.h"

#include <vital/exceptions/image.h>
#include <arrows/vxl/vil_image_memory.h>
#include <vxl_config.h>
#include <vil/vil_new.h>


namespace kwiver {
namespace arrows {
namespace vxl {


/// Constructor - from a vil_image_view_base
image_container
::image_container(const vil_image_view_base& d)
: data_(vil_new_image_view_base_sptr(d))
{
}


/// Constructor - convert base image container to vil
image_container
::image_container(const vital::image_container& image_cont)
{
  const vxl::image_container* vic =
      dynamic_cast<const vxl::image_container*>(&image_cont);
  if( vic )
  {
    this->data_ = vic->data_;
  }
  else
  {
    this->data_ = vital_to_vxl(image_cont.get_image());
  }
}


/// The size of the image data in bytes
/**
 * This size includes all allocated image memory,
 * which could be larger than width*height*depth.
 */
size_t
image_container
::size() const
{
  if( !data_ )
  {
    return 0;
  }
  switch (vil_pixel_format_component_format(data_->pixel_format()))
  {
#define CONVERT_CASE( F ) \
    case F: \
    { \
      typedef vil_pixel_format_type_of<F >::component_type pix_t; \
      vil_image_view<pix_t> img(data_); \
      if( !img.memory_chunk() ) \
      { \
        return 0; \
      } \
      return img.memory_chunk()->size(); \
    }
    CONVERT_CASE( VIL_PIXEL_FORMAT_BYTE)
    CONVERT_CASE( VIL_PIXEL_FORMAT_SBYTE)
#if VXL_HAS_INT_64
    CONVERT_CASE( VIL_PIXEL_FORMAT_UINT_64)
    CONVERT_CASE( VIL_PIXEL_FORMAT_INT_64)
#endif
    CONVERT_CASE( VIL_PIXEL_FORMAT_UINT_32)
    CONVERT_CASE( VIL_PIXEL_FORMAT_INT_32)
    CONVERT_CASE( VIL_PIXEL_FORMAT_UINT_16)
    CONVERT_CASE( VIL_PIXEL_FORMAT_INT_16)
    CONVERT_CASE( VIL_PIXEL_FORMAT_FLOAT)
    CONVERT_CASE( VIL_PIXEL_FORMAT_DOUBLE)
    CONVERT_CASE( VIL_PIXEL_FORMAT_BOOL)
#undef CONVERT_CASE
    default:
      return 0;
  }
  return 0;
}


/// Convert a VXL vil_image_view to a VITAL image
vital::image
image_container
::vxl_to_vital(const vil_image_view_base& img)
{
  switch (vil_pixel_format_component_format(img.pixel_format()))
  {
#define CONVERT_CASE( F ) \
    case F: \
    { \
      typedef vil_pixel_format_type_of<F >::component_type pix_t; \
      vil_image_view<pix_t> img_t(img); \
      vital::image_memory_sptr memory = vxl::vxl_to_vital(img_t.memory_chunk()); \
      return vital::image_of<pix_t>(memory, img_t.top_left_ptr(), \
                                    img_t.ni(), img_t.nj(), img_t.nplanes(), \
                                    img_t.istep(), img_t.jstep(), img_t.planestep()); \
    }
    CONVERT_CASE( VIL_PIXEL_FORMAT_BYTE)
    CONVERT_CASE( VIL_PIXEL_FORMAT_SBYTE)
#if VXL_HAS_INT_64
    CONVERT_CASE( VIL_PIXEL_FORMAT_UINT_64)
    CONVERT_CASE( VIL_PIXEL_FORMAT_INT_64)
#endif
    CONVERT_CASE( VIL_PIXEL_FORMAT_UINT_32)
    CONVERT_CASE( VIL_PIXEL_FORMAT_INT_32)
    CONVERT_CASE( VIL_PIXEL_FORMAT_UINT_16)
    CONVERT_CASE( VIL_PIXEL_FORMAT_INT_16)
    CONVERT_CASE( VIL_PIXEL_FORMAT_FLOAT)
    CONVERT_CASE( VIL_PIXEL_FORMAT_DOUBLE)
    CONVERT_CASE( VIL_PIXEL_FORMAT_BOOL)
#undef CONVERT_CASE
    default:
      throw vital::image_type_mismatch_exception("kwiver::arrows::vxl::image_container::vxl_to_vital(const vil_image_view_base&)");
  }
  return vital::image();
}


namespace
{

template <typename T>
inline
vil_image_view_base_sptr
make_vil_image_view(const vital::image& img)
{
  vil_memory_chunk_sptr chunk = vital_to_vxl(img.memory());
  return new vil_image_view<T>(chunk, reinterpret_cast<const T*>(img.first_pixel()),
                              static_cast<unsigned int>(img.width()),
                              static_cast<unsigned int>(img.height()),
                              static_cast<unsigned int>(img.depth()),
                              img.w_step(), img.h_step(), img.d_step());
}

}


/// Convert a VITAL image to a VXL vil_image_view
vil_image_view_base_sptr
image_container
::vital_to_vxl(const vital::image& img)
{
  const vital::image_pixel_traits& pt = img.pixel_traits();
  switch (pt.num_bytes)
  {
    case 1:
      switch (pt.type)
      {
        case vital::image_pixel_traits::BOOL:
          return make_vil_image_view<bool>(img);
        case vital::image_pixel_traits::UNSIGNED:
          return make_vil_image_view<vxl_byte>(img);
        case vital::image_pixel_traits::SIGNED:
          return make_vil_image_view<vxl_sbyte>(img);
        default:
          break;
      }

    case 2:
      switch (pt.type)
      {
        case vital::image_pixel_traits::UNSIGNED:
          return make_vil_image_view<vxl_uint_16>(img);
        case vital::image_pixel_traits::SIGNED:
          return make_vil_image_view<vxl_int_16>(img);
        default:
          break;
      }

    case 4:
      switch (pt.type)
      {
        case vital::image_pixel_traits::UNSIGNED:
          return make_vil_image_view<vxl_uint_32>(img);
        case vital::image_pixel_traits::SIGNED:
          return make_vil_image_view<vxl_int_32>(img);
        case vital::image_pixel_traits::FLOAT:
          return make_vil_image_view<float>(img);
        default:
          break;
      }

    case 8:
      switch (pt.type)
      {
#if VXL_HAS_INT_64
        case vital::image_pixel_traits::UNSIGNED:
          return make_vil_image_view<vxl_uint_64>(img);
        case vital::image_pixel_traits::SIGNED:
          return make_vil_image_view<vxl_int_64>(img);
#endif
        case vital::image_pixel_traits::FLOAT:
          return make_vil_image_view<double>(img);
        default:
          break;
      }
  }
  throw vital::image_type_mismatch_exception("kwiver::arrows::vxl::image_container::vital_to_vxl(const image&)");
  return vil_image_view_base_sptr();
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
