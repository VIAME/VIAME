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
 * \brief VXL image_io implementation
 */

#include "image_io.h"

#include <vital/logger/logger.h>
#include <vital/io/eigen_io.h>
#include <vital/types/vector.h>
#include <vital/exceptions/image.h>

#include <arrows/vxl/image_container.h>

#include <vil/vil_convert.h>
#include <vil/vil_load.h>
#include <vil/vil_save.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace vxl {

namespace
{

/// Helper function to convert images based on configuration
template <typename inP, typename outP>
void
convert_image_helper(const vil_image_view<inP>& src,
                     vil_image_view<outP>& dest,
                     bool force_byte, bool auto_stretch,
                     bool manual_stretch, const vector_2d& intensity_range)
{
  vil_image_view<double> temp;
  // The maximum value is extended by almost one such that dest_maxv still truncates
  // to the outP maximimum value after casting.  The purpose for this is to more evenly
  // distribute the values across the dynamic range.
  const double almost_one = 1 - 1e-6;
  double dest_minv = static_cast<double>(std::numeric_limits<outP>::min());
  double dest_maxv = static_cast<double>(std::numeric_limits<outP>::max()) + almost_one;
  if( !std::numeric_limits<outP>::is_integer )
  {
    dest_minv = outP(0);
    dest_maxv = outP(1);
  }
  if( auto_stretch )
  {
    vil_convert_stretch_range(src, temp, dest_minv, dest_maxv);
    vil_convert_cast(temp, dest);
  }
  else if( manual_stretch )
  {
    inP minv = static_cast<inP>(intensity_range[0]);
    inP maxv = static_cast<inP>(intensity_range[1]);
    vil_convert_stretch_range_limited(src, temp, minv, maxv, dest_minv, dest_maxv);
    vil_convert_cast(temp, dest);
  }
  else
  {
    vil_convert_cast(src, dest);
  }
}


/// Helper function to convert images based on configuration - specialized for byte output
template <typename inP>
void
convert_image_helper(const vil_image_view<inP>& src,
                     vil_image_view<vxl_byte>& dest,
                     bool force_byte, bool auto_stretch,
                     bool manual_stretch, const vector_2d& intensity_range)
{
  if( auto_stretch )
  {
    vil_convert_stretch_range(src, dest);
  }
  else if( manual_stretch )
  {
    inP minv = static_cast<inP>(intensity_range[0]);
    inP maxv = static_cast<inP>(intensity_range[1]);
    vil_convert_stretch_range_limited(src, dest, minv, maxv);
  }
  else
  {
    vil_convert_cast(src, dest);
  }
}


/// Helper function to convert images based on configuration - specialization for bool
template <typename outP>
void
convert_image_helper(const vil_image_view<bool>& src,
                     vil_image_view<outP>& dest,
                     bool force_byte, bool auto_stretch,
                     bool manual_stretch, const vector_2d& intensity_range)
{
  // special case for bool because manual stretching limits do not
  // make sense and trigger compiler warnings on some platforms.
  if( auto_stretch || manual_stretch )
  {
    vil_convert_stretch_range(src, dest);
  }
  else
  {
    vil_convert_cast(src, dest);
  }
}


/// Helper function to convert images based on configuration - resolve specialization ambiguity
void
convert_image_helper(const vil_image_view<bool>& src,
                     vil_image_view<vxl_byte>& dest,
                     bool force_byte, bool auto_stretch,
                     bool manual_stretch, const vector_2d& intensity_range)
{
  convert_image_helper<vxl_byte>(src, dest, force_byte, auto_stretch, manual_stretch, intensity_range);
}


/// Helper function to convert images based on configuration - specialization for bool/bool
void
convert_image_helper(const vil_image_view<bool>& src,
                     vil_image_view<bool>& dest,
                     bool force_byte, bool auto_stretch,
                     bool manual_stretch, const vector_2d& intensity_range)
{
  // special case for bool because stretch does not make sense for bool to bool conversion
  dest = src;
}


}



/// Private implementation class
class image_io::priv
{
public:
  /// Constructor
  priv()
  : force_byte(false),
    auto_stretch(false),
    manual_stretch(false),
    intensity_range(0, 255),
    m_logger( vital::get_logger( "arrows.vxl.image_io" ) )
  {
  }

  template <typename inP, typename outP>
  void convert_image(const vil_image_view<inP>& src,
                     vil_image_view<outP>& dest)
  {
    convert_image_helper(src, dest,
                         this->force_byte,
                         this->auto_stretch,
                         this->manual_stretch,
                         this->intensity_range);
  }

  bool force_byte;
  bool auto_stretch;
  bool manual_stretch;
  vector_2d intensity_range;

  vital::logger_handle_t m_logger;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/// Constructor
image_io
::image_io()
: d_(new priv)
{
}


/// Destructor
image_io
::~image_io()
{
}



/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
image_io
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = vital::algo::image_io::get_configuration();

  config->set_value("force_byte", d_->force_byte,
                    "When loading, convert the loaded data into a byte "
                    "(unsigned char) image regardless of the source data type. "
                    "Stretch the dynamic range according to the stretch options "
                    "before converting. When saving, convert to a byte image "
                    "before writing out the image");

  config->set_value("auto_stretch", d_->auto_stretch,
                    "Dynamically stretch the range of the input data such that "
                    "the minimum and maximum pixel values in the data map to "
                    "the minimum and maximum support values for that pixel "
                    "type, or 0.0 and 1.0 for floating point types.  If using "
                    "the force_byte option value map between 0 and 255. "
                    "Warning, this can result in brightness and constrast "
                    "varying between images.");

  config->set_value("manual_stretch", d_->manual_stretch,
                    "Manually stretch the range of the input data by "
                    "specifying the minimum and maximum values of the data "
                    "to map to the full byte range");
  if( d_->manual_stretch )
  {
    config->set_value("intensity_range", d_->intensity_range.transpose(),
                      "The range of intensity values (min, max) to stretch into "
                      "the byte range.  This is most useful when e.g. 12-bit "
                      "data is encoded in 16-bit pixels");
  }
  return config;
}


/// Set this algorithm's properties via a config block
void
image_io
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated vital::config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->force_byte = config->get_value<bool>("force_byte",
                                           d_->force_byte);
  d_->auto_stretch = config->get_value<bool>("auto_stretch",
                                              d_->auto_stretch);
  d_->manual_stretch = config->get_value<bool>("manual_stretch",
                                              d_->manual_stretch);
  d_->intensity_range = config->get_value<vector_2d>("intensity_range",
                                        d_->intensity_range.transpose());
}


/// Check that the algorithm's currently configuration is valid
bool
image_io
::check_configuration(vital::config_block_sptr config) const
{
  double auto_stretch = config->get_value<bool>("auto_stretch",
                                                d_->auto_stretch);
  double manual_stretch = config->get_value<bool>("manual_stretch",
                                                  d_->manual_stretch);
  if( auto_stretch && manual_stretch)
  {
    LOG_ERROR( d_->m_logger, "can not enable both manual and auto stretching");
    return false;
  }
  if( manual_stretch )
  {
    vector_2d range = config->get_value<vector_2d>("intensity_range",
                                        d_->intensity_range.transpose());
    if( range[0] >= range[1] )
    {
      LOG_ERROR( d_->m_logger, "stretching range minimum not less than maximum"
                <<" ("<<range[0]<<", "<<range[1]<<")");
      return false;
    }
  }
  return true;
}


/// Load image image from the file
image_container_sptr
image_io
::load_(const std::string& filename) const
{
  LOG_DEBUG( d_->m_logger, "Loading image from file: " << filename );

  vil_image_resource_sptr img_rsc = vil_load_image_resource(filename.c_str());

#define DO_CASE(T)                                                     \
  case T:                                                              \
    {                                                                  \
      typedef vil_pixel_format_type_of<T >::component_type pix_t;      \
      vil_image_view<pix_t> img_pix_t = img_rsc->get_view();           \
      if( d_->force_byte )                                             \
      {                                                                \
        vil_image_view<vxl_byte> img;                                  \
        d_->convert_image(img_pix_t, img);                             \
        return image_container_sptr(new vxl::image_container(img));    \
      }                                                                \
      else                                                             \
      {                                                                \
        vil_image_view<pix_t> img;                                     \
        d_->convert_image(img_pix_t, img);                             \
        return image_container_sptr(new vxl::image_container(img));    \
      }                                                                \
    }                                                                  \
    break;                                                             \

  switch (img_rsc->pixel_format())
  {
    DO_CASE(VIL_PIXEL_FORMAT_BOOL);
    DO_CASE(VIL_PIXEL_FORMAT_BYTE);
    DO_CASE(VIL_PIXEL_FORMAT_SBYTE);
    DO_CASE(VIL_PIXEL_FORMAT_UINT_16);
    DO_CASE(VIL_PIXEL_FORMAT_INT_16);
    DO_CASE(VIL_PIXEL_FORMAT_UINT_32);
    DO_CASE(VIL_PIXEL_FORMAT_INT_32);
    DO_CASE(VIL_PIXEL_FORMAT_UINT_64);
    DO_CASE(VIL_PIXEL_FORMAT_INT_64);
    DO_CASE(VIL_PIXEL_FORMAT_FLOAT);
    DO_CASE(VIL_PIXEL_FORMAT_DOUBLE);
#undef DO_CASE

  default:
    if( d_->auto_stretch )
    {
      // automatically stretch to fill the byte range using the
      // minimum and maximum pixel values
      vil_image_view<vxl_byte> img;
      img = vil_convert_stretch_range(vxl_byte(), img_rsc->get_view());
      return image_container_sptr(new vxl::image_container(img));
    }
    else if( d_->manual_stretch )
    {
      LOG_ERROR( d_->m_logger, "Unable to manually stretch pixel type: "
                << img_rsc->pixel_format());
      throw vital::image_type_mismatch_exception("kwiver::arrows::vxl::image_io::load_()");
    }
    else
    {
      vil_image_view<vxl_byte> img;
      img = vil_convert_cast(vxl_byte(), img_rsc->get_view());
      return image_container_sptr(new vxl::image_container(img));
    }
  }
  return image_container_sptr();
}


/// Save image image to a file
void
image_io
::save_(const std::string& filename,
       image_container_sptr data) const
{
  vil_image_view_base_sptr view =
    vxl::image_container::vital_to_vxl(data->get_image());

#define DO_CASE(T)                                                     \
  case T:                                                              \
    {                                                                  \
      typedef vil_pixel_format_type_of<T >::component_type pix_t;      \
      vil_image_view<pix_t> img_pix_t = view;                          \
      if( d_->force_byte )                                             \
      {                                                                \
        vil_image_view<vxl_byte> img;                                  \
        d_->convert_image(img_pix_t, img);                             \
        vil_save(img, filename.c_str());                               \
        return;                                                        \
      }                                                                \
      else                                                             \
      {                                                                \
        vil_image_view<pix_t> img;                                     \
        d_->convert_image(img_pix_t, img);                             \
        vil_save(img, filename.c_str());                               \
        return;                                                        \
      }                                                                \
    }                                                                  \
    break;                                                             \

  switch (view->pixel_format())
  {
    DO_CASE(VIL_PIXEL_FORMAT_BOOL);
    DO_CASE(VIL_PIXEL_FORMAT_BYTE);
    DO_CASE(VIL_PIXEL_FORMAT_SBYTE);
    DO_CASE(VIL_PIXEL_FORMAT_UINT_16);
    DO_CASE(VIL_PIXEL_FORMAT_INT_16);
    DO_CASE(VIL_PIXEL_FORMAT_UINT_32);
    DO_CASE(VIL_PIXEL_FORMAT_INT_32);
    DO_CASE(VIL_PIXEL_FORMAT_UINT_64);
    DO_CASE(VIL_PIXEL_FORMAT_INT_64);
    DO_CASE(VIL_PIXEL_FORMAT_FLOAT);
    DO_CASE(VIL_PIXEL_FORMAT_DOUBLE);
#undef DO_CASE

  default:
    if( d_->auto_stretch )
    {
      // automatically stretch to fill the byte range using the
      // minimum and maximum pixel values
      vil_image_view<vxl_byte> img;
      img = vil_convert_stretch_range(vxl_byte(), view);
      vil_save(img, filename.c_str());
      return;
    }
    else if( d_->manual_stretch )
    {
      LOG_ERROR( d_->m_logger, "Unable to manually stretch pixel type: "
                << view->pixel_format());
      throw vital::image_type_mismatch_exception("kwiver::arrows::vxl::image_io::save_()");
    }
    else
    {
      vil_image_view<vxl_byte> img;
      img = vil_convert_cast(vxl_byte(), view);
      vil_save(img, filename.c_str());
      return;
    }
  }
}

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver
