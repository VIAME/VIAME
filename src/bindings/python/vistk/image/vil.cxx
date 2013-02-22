/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <vistk/python/numpy/import.h>
#include <vistk/python/numpy/numpy_to_vil.h>
#include <vistk/python/numpy/registration.h>
#include <vistk/python/numpy/vil_to_numpy.h>

#include <vil/vil_image_view.h>

#include <boost/python/args.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/cstdint.hpp>

#include <Python.h>

/**
 * \file vil.cxx
 *
 * \brief Python bindings for \link vil_image_view\endlink.
 */

using namespace boost::python;

BOOST_PYTHON_MODULE(vil)
{
  vistk::python::import_numpy();

#define VIL_TYPES(call) \
  call(bool);           \
  call(uint8_t);        \
  call(float);          \
  call(double)

  vistk::python::register_memory_chunk();
  vistk::python::register_image_base();

#define REGISTER_IMAGE_TYPE(type)             \
  vistk::python::register_image_type<type>(); \
  vistk::python::register_type<vil_image_view<type> >(10)

  VIL_TYPES(REGISTER_IMAGE_TYPE);

#undef REGISTER_IMAGE_TYPE

  def("numpy_to_vil", &vistk::python::numpy_to_vil_base
    , (arg("numpy array"))
    , "Convert a NumPy array into a base vil image.");
  def("vil_to_numpy", &vistk::python::vil_base_to_numpy
    , (arg("numpy array"))
    , "Convert a NumPy array into a base vil image.");

#define DEFINE_CONVERSION_FUNCTIONS(type)                         \
  do                                                              \
  {                                                               \
    def("numpy_to_vil_" #type, &vistk::python::numpy_to_vil<type> \
      , (arg("numpy"))                                            \
      , "Convert a NumPy array into a " #type " vil image.");     \
    def("vil_to_numpy", &vistk::python::vil_to_numpy<type>        \
      , (arg("vil"))                                              \
      , "Convert a vil image into a NumPy array.");               \
  } while (false)

  VIL_TYPES(DEFINE_CONVERSION_FUNCTIONS);

#undef DEFINE_FUNCTIONS

#undef VIL_TYPES
}
