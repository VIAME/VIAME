/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "embedded_pipeline.h"
#include <sprokit/processes/adapters/embedded_pipeline.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE( embedded_pipeline, m )
{
  using description = viame::embedded_pipeline_description;
  py::class_< description >( m, "EmbeddedPipelineDescription" )
    .def_readonly( "pipeline_text", &description::pipeline_text )
    .def_readonly( "source_directory", &description::source_directory )
    .def_readonly( "input_names", &description::input_names )
    .def_readonly( "input_ports", &description::input_ports )
    .def_readonly( "output_ports", &description::output_ports )
    .def( "build", &description::build, py::arg( "pipeline" ),
          py::call_guard< py::gil_scoped_release >() );

  m.def( "prepare_embedded_pipeline",
    []( std::string const& path, std::vector< std::string > const& search_paths,
        std::optional< std::vector< std::string > > const& inputs,
        std::optional< std::vector< std::string > > const& outputs )
    {
      return viame::prepare_embedded_pipeline( path, { search_paths, inputs, outputs } );
    }, py::arg( "path" ), py::arg( "search_paths" ) = std::vector< std::string >{},
    py::arg( "inputs" ) = py::none(), py::arg( "outputs" ) = py::none(),
    py::call_guard< py::gil_scoped_release >() );
}
