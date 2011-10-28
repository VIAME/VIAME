/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_types/utility_types.h>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

/**
 * \file utility.cxx
 *
 * \brief Python bindings for utility pipeline types.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(utility)
{
  scope s;
  s.attr("t_timestamp") = vistk::utility_types::t_timestamp;
  s.attr("t_transform") = vistk::utility_types::t_transform;
  s.attr("t_image_to_image_homography") = vistk::utility_types::t_image_to_image_homography;
  s.attr("t_image_to_plane_homography") = vistk::utility_types::t_image_to_plane_homography;
  s.attr("t_plane_to_image_homography") = vistk::utility_types::t_plane_to_image_homography;
  s.attr("t_image_to_utm_homography") = vistk::utility_types::t_image_to_utm_homography;
  s.attr("t_utm_to_image_homography") = vistk::utility_types::t_utm_to_image_homography;
  s.attr("t_plane_to_utm_homography") = vistk::utility_types::t_plane_to_utm_homography;
  s.attr("t_utm_to_plane_homography") = vistk::utility_types::t_utm_to_plane_homography;
}
