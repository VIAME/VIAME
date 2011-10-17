/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PYTHON_HELPERS_PYTHON_CONVERT_ANY_H
#define VISTK_PYTHON_HELPERS_PYTHON_CONVERT_ANY_H

#include <boost/any.hpp>

#include <Python.h>

/**
 * \file python_convert_any.h
 *
 * \brief Helpers for working with boost::any in Python.
 */

namespace boost
{

namespace python
{

namespace converter
{

struct rvalue_from_python_stage1_data;

}

}

}

struct boost_any_to_object
{
  boost_any_to_object();
  ~boost_any_to_object();

  static void* convertible(PyObject* obj);
  static PyObject* convert(boost::any const& any);
  static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);
};

#endif // VISTK_PYTHON_PIPELINE_PYTHON_CONVERT_ANY_H
