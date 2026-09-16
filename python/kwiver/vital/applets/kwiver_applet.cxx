// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <python/kwiver/vital/applets/kwiver_applet_trampoline.txx>
#include <viame/algorithm_framework/applets/applet_context.h>
#include <viame/algorithm_framework/applets/kwiver_applet.h>

namespace py = pybind11;

namespace viame {

namespace tools {

namespace python {

void
kwiver_applet_binding( py::module& m )
{
  // Import required modules
  py::module::import( "kwiver.vital.config" );

  py::object const mod_pluggable = py::module::import( "kwiver.vital.plugins" );

  // Bind applet_context
  py::class_< viame::tools::applet_context >( m, "AppletContext" )
    .def( py::init<>() )
    .def_readwrite(
      "applet_name", &viame::tools::applet_context::m_applet_name,
      "Name of the applet as specified on command line" )
    .def_readwrite(
      "argv", &viame::tools::applet_context::m_argv,
      "Original arguments for the applet" )
    .def_readwrite(
      "skip_command_args_parsing",
      &viame::tools::applet_context::m_skip_command_args_parsing,
      "Flag to skip command line parsing" );

  // Bind kwiver_applet
  py::class_< viame::tools::kwiver_applet,
    viame::tools::kwiver_applet_sptr,
    viame::pluggable,
    viame::tools::python::kwiver_applet_trampoline<> >( m, "KwiverApplet" )
    .def( py::init<>() )
    .def(
      "run", &viame::tools::kwiver_applet::run,
      "Main entry point for the applet. Returns application return code." )
    .def(
      "add_command_options", &viame::tools::kwiver_applet::add_command_options,
      "Add command line options to the parser" )
    .def(
      "set_configuration", &viame::tools::kwiver_applet::set_configuration,
      py::arg( "config" ),
      "Set the applet configuration" )
    .def(
      "get_configuration", &viame::tools::kwiver_applet::get_configuration,
      "Get the applet configuration" )
    .def(
      "applet_name",
      []( const viame::tools::python::kwiver_applet_trampoline<>& self ){
        return self.public_applet_name();
      },
      "Get the applet name" )
    .def(
      "applet_args",
      []( const viame::tools::python::kwiver_applet_trampoline<>& self ){
        return self.public_applet_args();
      },
      "Get the original applet arguments" )
    .def(
      "wrap_text",
      []( viame::tools::python::kwiver_applet_trampoline<>& self,
          const std::string& text ){
        return self.public_wrap_text( text );
      },
      py::arg( "text" ),
      "Wrap text into a fixed width block" )
    .def_static(
      "find_configuration", &viame::tools::kwiver_applet::find_configuration,
      py::arg( "file_name" ),
      "Find and read a config file on the KWIVER config path" );
}

} // namespace python

} // namespace tools

} // namespace viame
