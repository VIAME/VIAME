/*ckwg +29
 * Copyright 2013-2016 by Kitware, Inc.
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
 * \brief OCV image_container implementation
 */

#include "image_container.h"

#include <arrows/ocv/mat_image_memory.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

image_container
::image_container(const cv::Mat& d, ColorMode cm)
  : data_(d)
{
  if(cm != RGB)
  {
    if ( data_.channels() == 3 )
    {
      cv::cvtColor(data_, data_, CV_BGR2RGB);
    }
    else if ( data_.channels() == 4 )
    {
      cv::cvtColor(data_, data_, CV_BGRA2RGBA);
    }
  }
}
/// Constructor - convert base image container to cv::Mat
image_container
::image_container(const vital::image_container& image_cont)
{
  // testing if image_cont is an ocv image container
  const ocv::image_container* oic =
      dynamic_cast<const ocv::image_container*>(&image_cont);
  if( oic )
  {
    this->data_ = oic->data_;
  }
  else
  {
    this->data_ = vital_to_ocv(image_cont.get_image());
  }
}


/// The size of the image data in bytes
size_t
image_container
::size() const
{
  return data_.rows * data_.step;
}


/// Convert an OpenCV cv::Mat to a VITAL image
image
image_container
::ocv_to_vital(const cv::Mat& img)
{
  // if the cv::Mat has reference counted memory then wrap it to keep a
  // counted reference too it.  If it doesn't own its memory, then the
  // vital image won't take ownership either
  image_memory_sptr memory;
#ifndef KWIVER_HAS_OPENCV_VER_3
  if ( !img.refcount )
#else
  if ( !img.u )
#endif
  {
    memory = std::make_shared<mat_image_memory>(img);
  }

  return image(memory, img.data,
               img.cols, img.rows, img.channels(),
               img.channels(), img.step1(), 1,
               ocv_to_vital(img.type()));
}


/// Convert an OpenCV cv::Mat type value to a vital::image_pixel_traits
vital::image_pixel_traits
image_container
::ocv_to_vital(int type)
{
  typedef vital::image_pixel_traits pixel_traits_t;
  switch(type % 8)
  {
    case CV_8U:
      return pixel_traits_t(vital::image_pixel_traits::UNSIGNED, 1);
    case CV_8S:
      return pixel_traits_t(vital::image_pixel_traits::SIGNED, 1);
    case CV_16U:
      return pixel_traits_t(vital::image_pixel_traits::UNSIGNED, 2);
    case CV_16S:
      return pixel_traits_t(vital::image_pixel_traits::SIGNED, 2);
    case CV_32S:
      return pixel_traits_t(vital::image_pixel_traits::SIGNED, 4);
    case CV_32F:
      return pixel_traits_t(vital::image_pixel_traits::FLOAT, 4);
    case CV_64F:
      return pixel_traits_t(vital::image_pixel_traits::FLOAT, 8);
    default:
      throw image_type_mismatch_exception("kwiver::arrows::ocv::image_container::ocv_to_vital(int)");
  }
}


/// Convert a VITAL image to an OpenCV cv::Mat
cv::Mat
image_container
::vital_to_ocv(const vital::image& img, image_container::ColorMode cm)
{
  // Find the matching OpenCV matrix type or throw and exception if there is no
  // compatible type
  const int cv_type = vital_to_ocv(img.pixel_traits());

  // cv::Mat is limited in the image data layouts and types that it supports.
  // Color channels must be interleaved (d_step==1) and the
  // step between columns must equal the number of channels (w_step==depth).
  // If the image does not have these properties we must allocate
  // a new cv::Mat and deep copy the data.  Otherwise, share memory.
  if( ( img.depth() == 1 || img.d_step() == 1 ) &&
      img.w_step() == static_cast<ptrdiff_t>(img.depth()) )
  {
    void * data_ptr = const_cast<void *>(img.first_pixel());
    cv::Mat out(static_cast<int>(img.height()), static_cast<int>(img.width()),
                CV_MAKETYPE(cv_type, static_cast<int>(img.depth())),
                data_ptr, img.h_step() * img.pixel_traits().num_bytes);

    // if this VITAL image is already wrapping cv::Mat allocated data,
    // then restore the original cv::Mat reference counter.
    image_memory_sptr memory = img.memory();
    if( mat_image_memory* mat_memory =
          dynamic_cast<mat_image_memory*>(memory.get()) )
    {
      // extract the existing reference counter from the VITAL wrapper
#ifndef KWIVER_HAS_OPENCV_VER_3
      out.refcount = mat_memory->get_ref_counter();
#else
      out.u = mat_memory->get_umatdata();
#endif
      out.addref();
    }
    // TODO use MatAllocator to share memory with image_memory
    if(cm == RGB || out.channels() == 1 )
    {
      return out;
    }
    else
    {
      cv::Mat bgr;
      if ( out.channels() == 3 )
      {
        cv::cvtColor(out, bgr, CV_RGB2BGR);
      }
      else if ( out.channels() == 4 )
      {
        cv::cvtColor(out, bgr, CV_RGBA2BGRA);
      }
      return bgr;
    }
  }

  // allocated a new cv::Mat
  cv::Mat out(static_cast<int>(img.height()), static_cast<int>(img.width()),
              CV_MAKETYPE(cv_type, static_cast<int>(img.depth())));
  // wrap the new image as a VITAL image (always a shallow copy)
  image new_img = ocv_to_vital(out);
  new_img.copy_from(img);

  if(cm == RGB || out.channels() == 1 )
  {
      return out;
  }
  else
  {
    cv::Mat bgr;
    if ( out.channels() == 3 )
    {
      cv::cvtColor(out, bgr, CV_RGB2BGR);
      return bgr;
    }
    if ( out.channels() == 4 )
    {
      cv::cvtColor(out, bgr, CV_RGBA2BGRA);
      return bgr;
    }
  }
  return out;
}


/// Convert a vital::image_pixel_traits to an OpenCV cv::Mat type integer
int
image_container
::vital_to_ocv(const vital::image_pixel_traits& pt)
{
  switch (pt.num_bytes)
  {
    case 1:
      if( pt.type == vital::image_pixel_traits::UNSIGNED )
      {
        return CV_8U;
      }
      if( pt.type == vital::image_pixel_traits::SIGNED )
      {
        return CV_8S;
      }
      break;

    case 2:
      if( pt.type == vital::image_pixel_traits::UNSIGNED )
      {
        return CV_16U;
      }
      if( pt.type == vital::image_pixel_traits::SIGNED )
      {
        return CV_16S;
      }
      break;

    case 4:
      if( pt.type == vital::image_pixel_traits::FLOAT )
      {
        return CV_32F;
      }
      if( pt.type == vital::image_pixel_traits::SIGNED )
      {
        return CV_32S;
      }
      break;

    case 8:
      if( pt.type == vital::image_pixel_traits::FLOAT )
      {
        return CV_64F;
      }
      break;

    default:
      break;
  }
  throw image_type_mismatch_exception("kwiver::arrows::ocv::image_container::vital_to_ocv(pixel_traits_t)");
}


/// Extract a cv::Mat from any image container
cv::Mat
image_container_to_ocv_matrix(const vital::image_container& img, image_container::ColorMode cm)
{
  cv::Mat result;
  if( const ocv::image_container* c =
          dynamic_cast<const ocv::image_container*>(&img) )
  {
    if(cm == image_container::RGB)
    {
      return c->get_Mat();
    }
    result = c->get_Mat().clone();
  }
  else
  {
    return ocv::image_container::vital_to_ocv(img.get_image(), cm);
  }
  if(cm == image_container::BGR)
  {
    if ( result.channels() == 3 )
    {
      cv::cvtColor(result, result, CV_RGB2BGR);
    }
    else if ( result.channels() == 4 )
    {
      cv::cvtColor(result, result, CV_RGBA2BGRA);
    }
  }
  return result;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
