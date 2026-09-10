// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <viame/pipeline_framework/load_pipe_exception.h>
#include <viame/pipeline_framework/adapters/embedded_pipeline.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <memory>

using namespace pybind11;

namespace kwiver {

namespace sprokit {

namespace python {

// Publicist class to access protected methods
class wrap_embedded_pipeline
  : public embedded_pipeline
{
public:
  using embedded_pipeline::connect_input_adapter;
  using embedded_pipeline::connect_output_adapter;
  using embedded_pipeline::update_config;
};

// Trampoline class to allow us to use virtual methods
class embedded_pipeline_trampoline
  : public embedded_pipeline
{
public:
  using embedded_pipeline::embedded_pipeline;

  bool connect_input_adapter() override;
  bool connect_output_adapter() override;
  void update_config( vital::config_block_sptr config ) override;
};

void build_pipeline(
  embedded_pipeline& self,
  vital::path_t const& desc_file,
  std::string const& def_dir = "" );

} // namespace python

} // namespace sprokit

} // namespace kwiver

namespace ksp = kwiver::sprokit::python;

PYBIND11_MODULE( embedded_pipeline, m )
{
  class_< kwiver::embedded_pipeline,
    std::shared_ptr< kwiver::embedded_pipeline >,
    ksp::embedded_pipeline_trampoline > ep( m, "EmbeddedPipeline" );
  ep.def( init<>() )
    .def(
      "build_pipeline", &ksp::build_pipeline,
      call_guard< pybind11::gil_scoped_release >(),
      arg( "desc_file" ),
      arg( "def_dir" ) = "" )
    .def(
      "send", &kwiver::embedded_pipeline::send,
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "send_end_of_input", &kwiver::embedded_pipeline::send_end_of_input,
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "receive", &kwiver::embedded_pipeline::receive,
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "full", &kwiver::embedded_pipeline::full,
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "empty", &kwiver::embedded_pipeline::empty,
      call_guard< pybind11::gil_scoped_release >() )
    .def( "at_end", &kwiver::embedded_pipeline::at_end )
    .def(
      "start", &kwiver::embedded_pipeline::start,
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "wait", &kwiver::embedded_pipeline::wait,
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "stop", &kwiver::embedded_pipeline::stop,
      call_guard< pybind11::gil_scoped_release >() )
    .def( "input_port_names", &kwiver::embedded_pipeline::input_port_names )
    .def( "output_port_names", &kwiver::embedded_pipeline::output_port_names )
    .def(
      "input_adapter_connected",
      &ksp::wrap_embedded_pipeline::input_adapter_connected )
    .def(
      "output_adapter_connected",
      &ksp::wrap_embedded_pipeline::output_adapter_connected )
    .def(
      "connect_input_adapter",
      static_cast< bool ( kwiver::embedded_pipeline::* )() >(
        &ksp::wrap_embedded_pipeline::connect_input_adapter ),
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "connect_output_adapter",
      static_cast< bool ( kwiver::embedded_pipeline::* )() >(
        &ksp::wrap_embedded_pipeline::connect_output_adapter ),
      call_guard< pybind11::gil_scoped_release >() )
    .def(
      "update_config",
      static_cast< void ( kwiver::embedded_pipeline::* )(
        kwiver::vital::config_block_sptr ) >(
        &ksp::wrap_embedded_pipeline::update_config ),
      call_guard< pybind11::gil_scoped_release >() )
  ;
  ep.doc() =
    R"(
        Python bindings for kwiver::embedded_pipeline

        Example:
            >>> from kwiver.sprokit.adapters import adapter_data_set, embedded_pipeline
            >>> import tempfile as tf, os
            >>> # Write a basic pipeline to our tempfile. Disable deletion on closing
            >>> # On Windows, the C++ process won't be able to access if still open in Python.
            >>> # We'll write to the file, then close it, so the C++ process can read. Then delete.
            >>> fp = tf.NamedTemporaryFile(mode="w+", delete=False)
            >>> fp.writelines(["process ia  :: input_adapter",
            >>>               "\nprocess oa  :: output_adapter",
            >>>               "\nconnect from ia.port1  to  oa.port2"])
            >>> fp.flush(); fp.close()
            >>>
            >>> ep = embedded_pipeline.EmbeddedPipeline()
            >>> ep.build_pipeline(fp.name)
            >>> assert list(ep.input_port_names()) == ["port1"]
            >>> assert list(ep.output_port_names()) == ["port2"]
            >>>
            >>> # Now lets run it
            >>> ep.start()
            >>> ads = adapter_data_set.AdapterDataSet.create()
            >>> ads["port1"] = 5
            >>> ep.send(ads)
            >>>
            >>> # All done, send end of input
            >>> ep.send_end_of_input()
            >>>
            >>> while True:
            >>>     ods = ep.receive()
            >>>     if not ods.is_end_of_data():
            >>>         assert ods["port2"] == 5
            >>>         break
            >>> os.remove(fp.name)
        )";
}

namespace kwiver {

namespace sprokit {

namespace python {

bool
embedded_pipeline_trampoline
::connect_input_adapter()
{
  PYBIND11_OVERLOAD(
    bool,
    embedded_pipeline,
    connect_input_adapter,
  );
}

bool
embedded_pipeline_trampoline
::connect_output_adapter()
{
  PYBIND11_OVERLOAD(
    bool,
    embedded_pipeline,
    connect_output_adapter,
  );
}

void
embedded_pipeline_trampoline
::update_config( vital::config_block_sptr config )
{
  PYBIND11_OVERLOAD(
    void,
    embedded_pipeline,
    update_config,
    config
  );
}

void
build_pipeline(
  embedded_pipeline& self,
  vital::path_t const& desc_file,
  std::string const& def_dir )
{
  std::ifstream desc_stream( desc_file );
  if( !desc_stream )
  {
    throw ::sprokit::file_no_exist_exception( desc_file );
  }
  self.build_pipeline( desc_stream, def_dir );
}

} // namespace python

} // namespace sprokit

} // namespace kwiver
