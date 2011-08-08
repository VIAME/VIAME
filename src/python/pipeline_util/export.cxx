/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline_util/export_dot.h>
#include <vistk/pipeline_util/export_dot_exception.h>

#include <boost/python.hpp>

/**
 * \file export.cxx
 *
 * \brief Python bindings for exporting functions.
 */

using namespace boost::python;

static void dot_translator(vistk::export_dot_exception const& e);

BOOST_PYTHON_MODULE(export_)
{
  register_exception_translator<
    vistk::export_dot_exception>(dot_translator);

  def("export_dot", &vistk::export_dot);
}

void
dot_translator(vistk::export_dot_exception const& e)
{
  PyErr_SetString(PyExc_RuntimeError, e.what());
}
