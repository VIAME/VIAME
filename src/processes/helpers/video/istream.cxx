/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "istream.h"

#include <processes/helpers/image/pixtypes.h>

#include <vistk/pipeline/datum.h>

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

#include <vidl/vidl_config.h>
#include <vidl/vidl_convert.h>
#include <vidl/vidl_pixel_format.h>

#include <vidl/vidl_image_list_istream.h>
#if VIDL_HAS_FFMPEG
#include <vidl/vidl_ffmpeg_istream.h>
#endif
#if VIDL_HAS_DSHOW
#include <vidl/vidl_dshow_file_istream.h>
#include <vidl/vidl_dshow_live_istream.h>
#endif
#if VIDL_HAS_DC1394
#include <vidl/vidl_dc1394_istream.h>
#endif
#if VIDL_HAS_VIDEODEV
#include <vidl/vidl_v4l_istream.h>
#endif
#if VIDL_HAS_VIDEODEV2
#include <vidl/vidl_v4l2_device.h>
#include <vidl/vidl_v4l2_istream.h>
#endif

#include <vil/vil_image_view.h>

/**
 * \file istream.cxx
 *
 * \brief Implementations of functions to help reading videos from a file.
 */

namespace vistk
{

template <typename PixType>
static datum_t istream_read(vidl_pixel_color color, istream_t const& istream);

static istream_impl_t const& glob_impl();
#if VIDL_HAS_FFMPEG
static istream_impl_t const& ffmpeg_impl();
#endif
#if VIDL_HAS_DSHOW
static istream_impl_t const& dshow_file_impl();
static istream_impl_t const& dshow_live_impl();
#endif
#if VIDL_HAS_DC1394
static istream_impl_t const& dc1394_impl();
#endif
#if VIDL_HAS_VIDEODEV
static istream_impl_t const& v4l_impl();
#endif
#if VIDL_HAS_VIDEODEV2
static istream_impl_t const& v4l2_impl();
#endif

istream_impl_t const&
default_istream_impl()
{
#if VIDL_HAS_FFMPEG
  return ffmpeg_impl();
#else
  return glob_impl();
#endif
}

istream_impls_t
known_istream_impls()
{
  istream_impls_t impls;

  impls.insert(glob_impl());
#if VIDL_HAS_FFMPEG
  impls.insert(ffmpeg_impl());
#endif
#if VIDL_HAS_DSHOW
  impls.insert(dshow_file_impl());
  impls.insert(dshow_live_impl());
#endif
#if VIDL_HAS_DC1394
  impls.insert(dc1394_impl());
#endif
#if VIDL_HAS_VIDEODEV
  impls.insert(v4l_impl());
#endif
#if VIDL_HAS_VIDEODEV2
  impls.insert(v4l2_impl());
#endif

  return impls;
}

istream_t
istream_for_impl(istream_impl_t const& impl, path_t const& path)
{
  path_t::string_type const& pstr = path.native();
  std::string const str(pstr.begin(), pstr.end());

  if (impl == glob_impl())
  {
    return boost::make_shared<vidl_image_list_istream>(str);
  }
#if VIDL_HAS_FFMPEG
  else if (impl == ffmpeg_impl())
  {
    return boost::make_shared<vidl_ffmpeg_istream>(str);
  }
#endif
#if VIDL_HAS_DSHOW
  else if (impl == dshow_file_impl())
  {
    return boost::make_shared<vidl_dshow_file_istream>(str);
  }
  else if (impl == dshow_live_impl())
  {
    return boost::make_shared<vidl_dshow_live_istream>(str);
  }
#endif
#if VIDL_HAS_DC1394
  else if (impl == dc1394_impl())
  {
    return boost::make_shared<vidl_dc1394_istream>();
  }
#endif
#if VIDL_HAS_VIDEODEV
  else if (impl == v4l_impl())
  {
    return boost::make_shared<vidl_v4l_istream>(str);
  }
#endif
#if VIDL_HAS_VIDEODEV2
  else if (impl == v4l2_impl())
  {
    vidl_v4l2_device device(str.c_str());

    return boost::make_shared<vidl_v4l2_istream>(boost::ref(device));
  }
#endif

  /// \todo Log warning that the given \p impl isn't known.

  return istream_t();
}

istream_read_func_t
istream_read_for_pixtype(pixtype_t const& pixtype, pixfmt_t const& pixfmt)
{
  vidl_pixel_color color;

  if (pixfmt == pixfmts::pixfmt_rgb())
  {
    color = VIDL_PIXEL_COLOR_RGB;
  }
  else if (pixfmt == pixfmts::pixfmt_rgba())
  {
    color = VIDL_PIXEL_COLOR_RGBA;
  }
  else if (pixfmt == pixfmts::pixfmt_yuv())
  {
    color = VIDL_PIXEL_COLOR_YUV;
  }
  else if (pixfmt == pixfmts::pixfmt_gray())
  {
    color = VIDL_PIXEL_COLOR_MONO;
  }
  else
  {
    /// \todo Log a warning that the pixel format is unsupported.
    return istream_read_func_t();
  }

  if (pixtype == pixtypes::pixtype_bool())
  {
    return boost::bind(&istream_read<bool>, color, _1);
  }
  else if (pixtype == pixtypes::pixtype_byte())
  {
    return boost::bind(&istream_read<uint8_t>, color, _1);
  }
  else if (pixtype == pixtypes::pixtype_float())
  {
    return boost::bind(&istream_read<float>, color, _1);
  }
  else if (pixtype == pixtypes::pixtype_double())
  {
    return boost::bind(&istream_read<double>, color, _1);
  }

  return istream_read_func_t();
}

template <typename PixType>
datum_t
istream_read(vidl_pixel_color color, istream_t const& istream)
{
  typedef vil_image_view<PixType> image_t;
  typedef vidl_frame_sptr frame_t;

  if (!istream->advance())
  {
    return datum::complete_datum();
  }

  if (!istream->is_valid())
  {
    static datum::error_t const err_string = datum::error_t("The input stream is invalid");

    return datum::error_datum(err_string);
  }

  frame_t const frame = istream->current_frame();

  if (!frame)
  {
    /// \bug Skip here instead?
    static datum::error_t const err_string = datum::error_t("An invalid frame was encountered");

    return datum::error_datum(err_string);
  }

  image_t img;

  if (!vidl_convert_to_view(*frame, img, color))
  {
    /// \bug Skip here instead?
    static datum::error_t const err_string = datum::error_t("Unable to convert frame into an image");

    return datum::error_datum(err_string);
  }

  if (!img)
  {
    /// \bug Skip here instead?
    static datum::error_t const err_string = datum::error_t("An empty image was created");

    return datum::error_datum(err_string);
  }

  return datum::new_datum(img);
}

istream_impl_t const&
glob_impl()
{
  static istream_impl_t impl = istream_impl_t("glob");

  return impl;
}

#if VIDL_HAS_FFMPEG
istream_impl_t const&
ffmpeg_impl()
{
  static istream_impl_t impl = istream_impl_t("ffmpeg");

  return impl;
}
#endif

#if VIDL_HAS_DSHOW
istream_impl_t const&
dshow_file_impl()
{
  static istream_impl_t impl = istream_impl_t("dshow_file");

  return impl;
}

istream_impl_t const&
dshow_live_impl()
{
  static istream_impl_t impl = istream_impl_t("dshow_live");

  return impl;
}
#endif

#if VIDL_HAS_DC1394
istream_impl_t const&
dc1394_impl()
{
  static istream_impl_t impl = istream_impl_t("dc1394");

  return impl;
}
#endif

#if VIDL_HAS_VIDEODEV
istream_impl_t const&
v4l_impl()
{
  static istream_impl_t impl = istream_impl_t("v4l");

  return impl;
}
#endif

#if VIDL_HAS_VIDEODEV2
istream_impl_t const&
v4l2_impl()
{
  static istream_impl_t impl = istream_impl_t("v4l2");

  return impl;
}
#endif

}
