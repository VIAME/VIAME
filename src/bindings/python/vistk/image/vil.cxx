/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/python/any_conversion/prototypes.h>
#include <vistk/python/any_conversion/registration.h>

#include <vistk/python/numpy/import.h>
#include <vistk/python/numpy/registration.h>

#include <boost/cstdint.hpp>

#include <Python.h>

/**
 * \file vil.cxx
 *
 * \brief Python bindings for \link vil_image_view\endlink.
 */

using namespace boost::python;

static pyimport_return_t import_numpy();

BOOST_PYTHON_MODULE(vil)
{
  vistk::python::import_numpy();

  vistk::python::register_memory_chunk();
  vistk::python::register_image_base();
  vistk::python::register_image_type();

  vistk::python::register_type<vil_image_view<bool> >(10);
  vistk::python::register_type<vil_image_view<uint8_t> >(10);
  vistk::python::register_type<vil_image_view<float> >(10);
  vistk::python::register_type<vil_image_view<double> >(10);
}
