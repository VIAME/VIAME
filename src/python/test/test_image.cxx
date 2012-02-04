/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

#include <vil/vil_image_view.h>

/**
 * \file test_image.cxx
 *
 * \brief Python bindings for image testing pipelines.
 */

using namespace boost::python;

template <typename T>
static vil_image_view<T> make_image(size_t width, size_t height, size_t planes);
template <typename T>
static size_t take_image(vil_image_view<T> const& img);

BOOST_PYTHON_MODULE(test_image)
{
#define DEFINE_FUNCTIONS(type)                       \
  do                                                 \
  {                                                  \
    def("make_image_" #type, &make_image<type>       \
      , (arg("width"), arg("height"), arg("planes")) \
      , "Create a " #type " image.");                \
    def("take_image_" #type, &take_image<type>       \
      , (arg("image")) \
      , "Take a " #type " image and return its size.");                \
  } while (false)

  DEFINE_FUNCTIONS(bool);
  DEFINE_FUNCTIONS(uint8_t);
  DEFINE_FUNCTIONS(float);
  DEFINE_FUNCTIONS(double);

#undef DEFINE_FUNCTIONS
}

template <typename T>
vil_image_view<T>
make_image(size_t width, size_t height, size_t planes)
{
  return vil_image_view<T>(width, height, planes);
}

template <typename T>
size_t
take_image(vil_image_view<T> const& img)
{
  size_t const ni = img.ni();
  size_t const nj = img.nj();
  size_t const np = img.nplanes();

  return (ni * nj * np);
}
