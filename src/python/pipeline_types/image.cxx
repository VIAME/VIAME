/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_types/image_types.h>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

/**
 * \file image.cxx
 *
 * \brief Python bindings for image pipeline types.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(image)
{
  scope s;
  s.attr("t_bitmask") = vistk::image_types::t_bitmask;
  s.attr("t_bytemask") = vistk::image_types::t_bytemask;
  s.attr("t_byte_grayscale") = vistk::image_types::t_byte_grayscale;
  s.attr("t_float_grayscale") = vistk::image_types::t_float_grayscale;
  s.attr("t_byte_rgb") = vistk::image_types::t_byte_rgb;
  s.attr("t_float_rgb") = vistk::image_types::t_float_rgb;
  s.attr("t_byte_bgr") = vistk::image_types::t_byte_bgr;
  s.attr("t_float_bgr") = vistk::image_types::t_float_bgr;
  s.attr("t_byte_rgba") = vistk::image_types::t_byte_rgba;
  s.attr("t_float_rgba") = vistk::image_types::t_float_rgba;
  s.attr("t_byte_bgra") = vistk::image_types::t_byte_bgra;
  s.attr("t_float_bgra") = vistk::image_types::t_float_bgra;
  s.attr("t_byte_yuv") = vistk::image_types::t_byte_yuv;
  s.attr("t_float_yuv") = vistk::image_types::t_float_yuv;
  s.attr("t_ipl") = vistk::image_types::t_ipl;
}
