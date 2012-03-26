/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/pystream.h>

#include <vistk/pipeline_util/export_dot.h>
#include <vistk/pipeline_util/export_dot_exception.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/exception_translator.hpp>

#include <boost/python.hpp>

#include <string>

/**
 * \file export.cxx
 *
 * \brief Python bindings for exporting functions.
 */

using namespace boost::python;

void export_dot(object const& stream, vistk::pipeline_t const pipe, std::string const& graph_name);

static void dot_translator(vistk::export_dot_exception const& e);

BOOST_PYTHON_MODULE(export_)
{
  register_exception_translator<
    vistk::export_dot_exception>(dot_translator);

  def("export_dot", &export_dot
    , (arg("stream"), arg("pipeline"), arg("name"))
    , "Writes the pipeline to the stream in dot format.");
}

void
dot_translator(vistk::export_dot_exception const& e)
{
  vistk::python::python_gil gil;

  (void)gil;

  PyErr_SetString(PyExc_RuntimeError, e.what());
}

void
export_dot(object const& stream, vistk::pipeline_t const pipe, std::string const& graph_name)
{
  vistk::python::python_gil gil;

  (void)gil;

  pyostream ostr(stream);

  return vistk::export_dot(ostr, pipe, graph_name);
}
