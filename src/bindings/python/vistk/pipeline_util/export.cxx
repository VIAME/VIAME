/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
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

BOOST_PYTHON_MODULE(export_)
{
  def("export_dot", &export_dot
    , (arg("stream"), arg("pipeline"), arg("name"))
    , "Writes the pipeline to the stream in dot format.");
}

void
export_dot(object const& stream, vistk::pipeline_t const pipe, std::string const& graph_name)
{
  vistk::python::python_gil const gil;

  (void)gil;

  pyostream ostr(stream);

  return vistk::export_dot(ostr, pipe, graph_name);
}
