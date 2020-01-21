/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sprokit/pipeline_util/pipeline_builder.h>
#include <sprokit/pipeline_util/pipe_bakery.h>
#include <sprokit/pipeline_util/pipe_bakery_exception.h>

#include <sprokit/pipeline/pipeline.h>
#include <sprokit/pipeline/process_factory.h>

#include <vital/bindings/python/vital/util/pybind11.h>
#include <sprokit/python/util/pystream.h>

#include <pybind11/pybind11.h>

#include <stdexcept>
#include <string>

/**
 * \file bake.cxx
 *
 * \brief Python bindings for baking pipelines.
 */

using namespace pybind11;

static sprokit::process::type_t cluster_info_type(sprokit::cluster_info_t const& self);
static sprokit::process::description_t cluster_info_description(sprokit::cluster_info_t const& self);
static sprokit::process_t cluster_info_create(sprokit::cluster_info_t const& self, kwiver::vital::config_block_sptr const& config);
static sprokit::process_t cluster_info_create_default(sprokit::cluster_info_t const& self);
static void register_cluster(sprokit::cluster_info_t const& info);
static sprokit::pipeline_t bake_pipe_file(std::string const& path);
static sprokit::pipeline_t bake_pipe(object stream);
static sprokit::cluster_info_t bake_cluster_file(std::string const& path);
static sprokit::cluster_info_t bake_cluster(object stream);

PYBIND11_MODULE(bake, m)
{
  class_<sprokit::cluster_info, sprokit::cluster_info_t>(m, "ClusterInfo"
    , "Information loaded from a cluster file.")
    .def("type", &cluster_info_type, call_guard<kwiver::vital::python::gil_scoped_release>())
    .def("description", &cluster_info_description, call_guard<kwiver::vital::python::gil_scoped_release>())
    .def("create", &cluster_info_create, call_guard<kwiver::vital::python::gil_scoped_release>()
      , (arg("config"))
      , "Create an instance of the cluster.")
    .def("create", &cluster_info_create_default, call_guard<kwiver::vital::python::gil_scoped_release>()
      , "Create an instance of the cluster.")
  ;

  m.def("register_cluster", &register_cluster, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("cluster_info"))
    , "Register a cluster with the registry.");

  m.def("bake_pipe_file", &bake_pipe_file, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("path"))
    , "Build a pipeline from a file.");
  m.def("bake_pipe", &bake_pipe, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("stream"))
    , "Build a pipeline from a stream.");
  m.def("bake_pipe_blocks", &sprokit::bake_pipe_blocks, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("blocks"))
    , "Build a pipeline from pipe blocks.");
  m.def("bake_cluster_file", &bake_cluster_file, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("path"))
    , "Build a cluster from a file.");
  m.def("bake_cluster", &bake_cluster, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("stream"))
    , "Build a cluster from a stream.");
  m.def("bake_cluster_blocks", &sprokit::bake_cluster_blocks, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("blocks"))
    , "Build a cluster from cluster blocks.");
  m.def("extract_configuration", &sprokit::extract_configuration, call_guard<kwiver::vital::python::gil_scoped_release>()
    , (arg("blocks"))
    , "Extract the configuration from pipe blocks.");
}


// ------------------------------------------------------------------
sprokit::process::type_t
cluster_info_type(sprokit::cluster_info_t const& self)
{
  return self->type;
}


// ------------------------------------------------------------------
sprokit::process::description_t
cluster_info_description(sprokit::cluster_info_t const& self)
{
  return self->description;
}


// ------------------------------------------------------------------
sprokit::process_t
cluster_info_create( sprokit::cluster_info_t const& self,
                     kwiver::vital::config_block_sptr const& config )
{
  sprokit::process_factory_func_t const& ctor = self->ctor;

  return ctor(config);
}


// ------------------------------------------------------------------
sprokit::process_t
cluster_info_create_default(sprokit::cluster_info_t const& self)
{
  kwiver::vital::config_block_sptr const conf = kwiver::vital::config_block::empty_config();

  return cluster_info_create(self, conf);
}


// ------------------------------------------------------------------
void
register_cluster(sprokit::cluster_info_t const& info)
{
  if (!info)
  {
    static std::string const reason = "A NULL cluster info was attempted to be registered";

    throw std::runtime_error(reason); // python compatible exception
  }

  sprokit::process::type_t const& type = info->type;
  sprokit::process::description_t const& description = info->description;
  sprokit::process_factory_func_t const& ctor = info->ctor;

  kwiver::vital::plugin_manager& vpm = kwiver::vital::plugin_manager::instance();
  sprokit::process::type_t derived_type = "python::";
  auto fact = vpm.add_factory( new sprokit::cpp_process_factory( derived_type + type, // derived type name string
                                                                 type, // name of the cluster/process
                                                                 ctor ) );

  fact->add_attribute( kwiver::vital::plugin_factory::PLUGIN_NAME, type )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_MODULE_NAME, "python-runtime-cluster" )
    .add_attribute( kwiver::vital::plugin_factory::PLUGIN_DESCRIPTION, description );
}


// ------------------------------------------------------------------
sprokit::pipeline_t
bake_pipe_file(std::string const& path)
{
  sprokit::pipeline_builder builder;
  builder.load_pipeline( path );
  return builder.pipeline();
}


// ------------------------------------------------------------------
sprokit::pipeline_t
bake_pipe(object stream)
{
  sprokit::python::pyistream istr(stream);
  sprokit::pipeline_builder builder;
  builder.load_pipeline( istr );
  return builder.pipeline();
}


// ------------------------------------------------------------------
sprokit::cluster_info_t
bake_cluster_file(std::string const& path)
{
  sprokit::pipeline_builder builder;
  builder.load_cluster( path );
  return builder.cluster_info();
}


// ------------------------------------------------------------------
sprokit::cluster_info_t
bake_cluster(object stream)
{
  sprokit::python::pyistream istr(stream);
  sprokit::pipeline_builder builder;
  builder.load_cluster( istr );
  return builder.cluster_info();
}
