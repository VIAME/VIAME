/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

#include <vil/vil_image_view.h>
#include <vil/vil_save.h>

/**
 * \file test_image.cxx
 *
 * \brief Python bindings for image testing pipelines.
 */

using namespace boost::python;

static vil_image_view_base_sptr make_image_base(size_t width, size_t height, size_t planes);
static size_t take_image_base(vil_image_view_base_sptr const& img);
static bool save_image_base(vil_image_view_base_sptr const& img, std::string const& path);
static vil_image_view_base_sptr pass_image_base(vil_image_view_base_sptr const& img);

template <typename T>
static vil_image_view<T> make_image(size_t width, size_t height, size_t planes);
template <typename T>
static size_t take_image(vil_image_view<T> const& img);
template <typename T>
static bool save_image(vil_image_view<T> const& img, std::string const& path);
template <typename T>
static vil_image_view<T> pass_image(vil_image_view<T> const& img);

BOOST_PYTHON_MODULE(test_image)
{
  def("make_image_base", &make_image_base
    , (arg("width"), arg("height"), arg("planes"))
    , "Create a base image.");
  def("take_image_base", &take_image_base
    , (arg("image"))
    , "Take a base image and return its size.");
  def("save_image_base", &save_image_base
    , (arg("image"), arg("path"))
    , "Take a base image and write it to a file.");
  def("pass_image_base", &pass_image_base
    , (arg("image"))
    , "Take a base image and return it.");

#define DEFINE_FUNCTIONS(type)                             \
  do                                                       \
  {                                                        \
    def("make_image_" #type, &make_image<type>             \
      , (arg("width"), arg("height"), arg("planes"))       \
      , "Create a " #type " image.");                      \
    def("take_image_" #type, &take_image<type>             \
      , (arg("image"))                                     \
      , "Take a " #type " image and return its size.");    \
    def("save_image_" #type, &save_image<type>             \
      , (arg("image"), arg("path"))                        \
      , "Take a " #type " image and write it to a file."); \
    def("pass_image_" #type, &pass_image<type>             \
      , (arg("image"))                                     \
      , "Take a " #type " image and return it.");          \
  } while (false)

  DEFINE_FUNCTIONS(bool);
  DEFINE_FUNCTIONS(uint8_t);
  DEFINE_FUNCTIONS(float);
  DEFINE_FUNCTIONS(double);

#undef DEFINE_FUNCTIONS
}

vil_image_view_base_sptr
make_image_base(size_t width, size_t height, size_t planes)
{
  typedef uint8_t base_type;

  return vil_image_view_base_sptr(new vil_image_view<base_type>(width, height, planes));
}

size_t
take_image_base(vil_image_view_base_sptr const& img)
{
  size_t const ni = img->ni();
  size_t const nj = img->nj();
  size_t const np = img->nplanes();

  return (ni * nj * np);
}

bool
save_image_base(vil_image_view_base_sptr const& img, std::string const& path)
{
  return vil_save(*img, path.c_str());
}

vil_image_view_base_sptr
pass_image_base(vil_image_view_base_sptr const& img)
{
  return img;
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

template <typename T>
bool
save_image(vil_image_view<T> const& img, std::string const& path)
{
  return vil_save(img, path.c_str());
}

template <typename T>
vil_image_view<T>
pass_image(vil_image_view<T> const& img)
{
  return img;
}
