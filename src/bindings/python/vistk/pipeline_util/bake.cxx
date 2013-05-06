/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/pystream.h>

#include <sprokit/pipeline_util/path.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process_registry.h>

#include <sprokit/python/util/python_gil.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/exception_translator.hpp>

#include <stdexcept>
#include <string>

/**
 * \file bake.cxx
 *
 * \brief Python bindings for baking pipelines.
 */

using namespace boost::python;

static sprokit::process::type_t cluster_info_type(sprokit::cluster_info_t const& self);
static sprokit::process_registry::description_t cluster_info_description(sprokit::cluster_info_t const& self);
static sprokit::process_t cluster_info_create(sprokit::cluster_info_t const& self, sprokit::config_t const& config);
static sprokit::process_t cluster_info_create_default(sprokit::cluster_info_t const& self);
static void register_cluster(sprokit::cluster_info_t const& info);
static sprokit::pipeline_t bake_pipe_file(std::string const& path);
static sprokit::pipeline_t bake_pipe(object stream, std::string const& inc_root);
static sprokit::cluster_info_t bake_cluster_file(std::string const& path);
static sprokit::cluster_info_t bake_cluster(object stream, std::string const& inc_root);

BOOST_PYTHON_MODULE(bake)
{
  class_<sprokit::cluster_info_t>("ClusterInfo"
    , "Information loaded from a cluster file."
    , no_init)
    .def("type", &cluster_info_type)
    .def("description", &cluster_info_description)
    .def("create", &cluster_info_create
      , (arg("config"))
      , "Create an instance of the cluster.")
    .def("create", &cluster_info_create_default
      , "Create an instance of the cluster.")
  ;

  def("register_cluster", &register_cluster
    , (arg("cluster_info"))
    , "Register a cluster with the registry.");

  def("bake_pipe_file", &bake_pipe_file
    , (arg("path"))
    , "Build a pipeline from a file.");
  def("bake_pipe", &bake_pipe
    , (arg("stream"), arg("inc_root") = std::string())
    , "Build a pipeline from a stream.");
  def("bake_pipe_blocks", &sprokit::bake_pipe_blocks
    , (arg("blocks"))
    , "Build a pipeline from pipe blocks.");
  def("bake_cluster_file", &bake_cluster_file
    , (arg("path"))
    , "Build a cluster from a file.");
  def("bake_cluster", &bake_cluster
    , (arg("stream"), arg("inc_root") = std::string())
    , "Build a cluster from a stream.");
  def("bake_cluster_blocks", &sprokit::bake_cluster_blocks
    , (arg("blocks"))
    , "Build a cluster from cluster blocks.");
  def("extract_configuration", &sprokit::extract_configuration
    , (arg("blocks"))
    , "Extract the configuration from pipe blocks.");
}

sprokit::process::type_t
cluster_info_type(sprokit::cluster_info_t const& self)
{
  return self->type;
}

sprokit::process_registry::description_t
cluster_info_description(sprokit::cluster_info_t const& self)
{
  return self->description;
}

sprokit::process_t
cluster_info_create(sprokit::cluster_info_t const& self, sprokit::config_t const& config)
{
  sprokit::process_ctor_t const& ctor = self->ctor;

  return ctor(config);
}

sprokit::process_t
cluster_info_create_default(sprokit::cluster_info_t const& self)
{
  sprokit::config_t const conf = sprokit::config::empty_config();

  return cluster_info_create(self, conf);
}

void
register_cluster(sprokit::cluster_info_t const& info)
{
  if (!info)
  {
    static std::string const reason = "A NULL cluster info was attempted to be registered";

    throw std::runtime_error(reason);
  }

  sprokit::process_registry_t const reg = sprokit::process_registry::self();

  sprokit::process::type_t const& type = info->type;
  sprokit::process_registry::description_t const& description = info->description;
  sprokit::process_ctor_t const& ctor = info->ctor;

  reg->register_process(type, description, ctor);
}

sprokit::pipeline_t
bake_pipe_file(std::string const& path)
{
  return sprokit::bake_pipe_from_file(sprokit::path_t(path));
}

sprokit::pipeline_t
bake_pipe(object stream, std::string const& inc_root)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  pyistream istr(stream);

  return sprokit::bake_pipe(istr, sprokit::path_t(inc_root));
}

sprokit::cluster_info_t
bake_cluster_file(std::string const& path)
{
  return sprokit::bake_cluster_from_file(sprokit::path_t(path));
}

sprokit::cluster_info_t
bake_cluster(object stream, std::string const& inc_root)
{
  sprokit::python::python_gil const gil;

  (void)gil;

  pyistream istr(stream);

  return sprokit::bake_cluster(istr, sprokit::path_t(inc_root));
}
