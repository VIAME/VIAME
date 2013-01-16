/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <python/helpers/pystream.h>

#include <vistk/pipeline_util/pipe_bakery.h>
#include <vistk/pipeline_util/pipe_bakery_exception.h>

#include <vistk/pipeline/pipeline.h>
#include <vistk/pipeline/process_registry.h>

#include <vistk/utilities/path.h>

#include <vistk/python/util/python_gil.h>

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

static vistk::process::type_t cluster_info_type(vistk::cluster_info_t const& self);
static vistk::process_registry::description_t cluster_info_description(vistk::cluster_info_t const& self);
static vistk::process_t cluster_info_create(vistk::cluster_info_t const& self, vistk::config_t const& config);
static vistk::process_t cluster_info_create_default(vistk::cluster_info_t const& self);
static void register_cluster(vistk::cluster_info_t const& info);
static vistk::pipeline_t bake_pipe_file(std::string const& path);
static vistk::pipeline_t bake_pipe(object stream, std::string const& inc_root);
static vistk::cluster_info_t bake_cluster_file(std::string const& path);
static vistk::cluster_info_t bake_cluster(object stream, std::string const& inc_root);

BOOST_PYTHON_MODULE(bake)
{
  class_<vistk::cluster_info_t>("ClusterInfo"
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
  def("bake_pipe_blocks", &vistk::bake_pipe_blocks
    , (arg("blocks"))
    , "Build a pipeline from pipe blocks.");
  def("bake_cluster_file", &bake_cluster_file
    , (arg("path"))
    , "Build a cluster from a file.");
  def("bake_cluster", &bake_cluster
    , (arg("stream"), arg("inc_root") = std::string())
    , "Build a cluster from a stream.");
  def("bake_cluster_blocks", &vistk::bake_cluster_blocks
    , (arg("blocks"))
    , "Build a cluster from cluster blocks.");
  def("extract_configuration", &vistk::extract_configuration
    , (arg("blocks"))
    , "Extract the configuration from pipe blocks.");
}

vistk::process::type_t
cluster_info_type(vistk::cluster_info_t const& self)
{
  return self->type;
}

vistk::process_registry::description_t
cluster_info_description(vistk::cluster_info_t const& self)
{
  return self->description;
}

vistk::process_t
cluster_info_create(vistk::cluster_info_t const& self, vistk::config_t const& config)
{
  vistk::process_ctor_t const& ctor = self->ctor;

  return ctor(config);
}

vistk::process_t
cluster_info_create_default(vistk::cluster_info_t const& self)
{
  vistk::config_t const conf = vistk::config::empty_config();

  return cluster_info_create(self, conf);
}

void
register_cluster(vistk::cluster_info_t const& info)
{
  if (!info)
  {
    static std::string const reason = "A NULL cluster info was attempted to be registered";

    throw std::runtime_error(reason);
  }

  vistk::process_registry_t const reg = vistk::process_registry::self();

  vistk::process::type_t const& type = info->type;
  vistk::process_registry::description_t const& description = info->description;
  vistk::process_ctor_t const& ctor = info->ctor;

  reg->register_process(type, description, ctor);
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

vistk::cluster_info_t
bake_cluster_file(std::string const& path)
{
  return vistk::bake_cluster_from_file(vistk::path_t(path));
}

vistk::cluster_info_t
bake_cluster(object stream, std::string const& inc_root)
{
  vistk::python::python_gil const gil;

  (void)gil;

  pyistream istr(stream);

  return vistk::bake_cluster(istr, vistk::path_t(inc_root));
}
