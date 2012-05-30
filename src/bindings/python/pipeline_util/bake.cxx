/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/pystream.h>

#include <vistk/pipeline_util/pipe_bakery.h>
#include <vistk/pipeline_util/pipe_bakery_exception.h>

#include <vistk/pipeline/pipeline.h>

#include <vistk/utilities/path.h>

#include <vistk/python/util/python_gil.h>

#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/exception_translator.hpp>

#include <string>

/**
 * \file bake.cxx
 *
 * \brief Python bindings for baking pipelines.
 */

using namespace boost::python;

static vistk::pipeline_t bake_pipe_file(std::string const& path);
static vistk::pipeline_t bake_pipe(object stream, std::string const& inc_root);

BOOST_PYTHON_MODULE(bake)
{
  def("bake_pipe_file", &bake_pipe_file
    , (arg("path"))
    , "Build a pipeline from a file.");
  def("bake_pipe", &bake_pipe
    , (arg("stream"), arg("inc_root") = std::string())
    , "Build a pipeline from a stream.");
  def("bake_pipe_blocks", &vistk::bake_pipe_blocks
    , (arg("blocks"))
    , "Build a pipeline from pipe blocks.");
  def("extract_configuration", &vistk::extract_configuration
    , (arg("blocks"))
    , "Extract the configuration from pipe blocks.");
}

vistk::pipeline_t
bake_pipe_file(std::string const& path)
{
  return vistk::bake_pipe_from_file(vistk::path_t(path));
}

vistk::pipeline_t
bake_pipe(object stream, std::string const& inc_root)
{
  vistk::python::python_gil const gil;

  (void)gil;

  pyistream istr(stream);

  return vistk::bake_pipe(istr, vistk::path_t(inc_root));
}
