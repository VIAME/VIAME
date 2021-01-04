// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <sprokit/pipeline/datum.h>
#include <sprokit/pipeline/edge.h>
#include <sprokit/pipeline/stamp.h>

#include <pybind11/stl_bind.h>

#include "python_wrappers.cxx"

/**
 * \file edge.cxx
 *
 * \brief Python bindings for \link sprokit::edge\endlink.
 */

using namespace pybind11;

namespace kwiver{
namespace sprokit{
namespace python{
static void push_datum(::sprokit::edge& self, wrap_edge_datum const& datum);
static wrap_edge_datum get_datum(::sprokit::edge& self);
static wrap_edge_datum peek_datum(::sprokit::edge& self, pybind11::size_t const& idx);
}
}
}
using namespace kwiver::sprokit::python;
PYBIND11_MODULE(edge, m)
{
  class_<wrap_edge_datum>(m, "EdgeDatum")
    .def(init<>())
    .def(init<sprokit::datum, wrap_stamp>())
    .def_readwrite("datum", &sprokit::edge_datum_t::datum)
    .def_readwrite("stamp", &sprokit::edge_datum_t::stamp)
    .def_property("datum", &wrap_edge_datum::get_datum, &wrap_edge_datum::set_datum)
    .def_property("stamp", &wrap_edge_datum::get_stamp, &wrap_edge_datum::set_stamp)
  ;
  bind_vector<std::vector<wrap_edge_datum> >(m, "EdgeData"
    , "A collection of data packets that may be passed through an edge.");

  class_<sprokit::edges_t>(m, "Edges"
    , "A collection of edges.")
    .def(pybind11::init<>());

  class_<sprokit::edge, sprokit::edge_t>(m, "Edge"
    , "A communication channel between processes.")
    .def(init<>())
    .def(init<kwiver::vital::config_block_sptr>())
    .def("makes_dependency", &sprokit::edge::makes_dependency
      , "Returns True if the edge implies a dependency from downstream on upstream.")
    .def("has_data", &sprokit::edge::has_data
      , "Returns True if the edge contains data, False otherwise.")
    .def("full_of_data", &sprokit::edge::full_of_data
      , "Returns True if the edge cannot hold anymore data, False otherwise.")
    .def("datum_count", &sprokit::edge::datum_count
      , "Returns the number of data packets within the edge.")
    .def("push_datum", &push_datum
      , (arg("datum"))
      , "Pushes a datum packet into the edge.")
    .def("get_datum", &get_datum
      , "Returns the next datum packet from the edge, removing it in the process.")
    .def("peek_datum", &peek_datum
      , (arg("index") = 0)
      , "Returns the next datum packet from the edge.")
    .def("pop_datum", &sprokit::edge::pop_datum
      , "Remove the next datum packet from the edge.")
    .def("set_upstream_process", &sprokit::edge::set_upstream_process
      , (arg("process"))
      , "Set the process which is feeding data into the edge.")
    .def("set_downstream_process", &sprokit::edge::set_downstream_process
      , (arg("process"))
      , "Set the process which is reading data from the edge.")
    .def("mark_downstream_as_complete", &sprokit::edge::mark_downstream_as_complete
      , "Indicate that the downstream process is complete.")
    .def("is_downstream_complete", &sprokit::edge::is_downstream_complete
      , "Returns True if the downstream process is complete, False otherwise.")
    .def_readonly_static("config_dependency", &sprokit::edge::config_dependency)
    .def_readonly_static("config_capacity", &sprokit::edge::config_capacity)
  ;
}

namespace kwiver{
namespace sprokit{
namespace python{
void
push_datum(::sprokit::edge& self, wrap_edge_datum const& datum)
{
  self.push_datum((::sprokit::edge_datum_t) datum);
}

wrap_edge_datum
get_datum(::sprokit::edge& self)
{
  ::sprokit::edge_datum_t datum = self.get_datum();
  wrap_edge_datum datum_p(*(datum.datum), wrap_stamp(datum.stamp));
  return datum_p;
}

wrap_edge_datum
peek_datum(::sprokit::edge& self, pybind11::size_t const& idx)
{
  ::sprokit::edge_datum_t datum = self.peek_datum(idx);
  wrap_edge_datum datum_p(*(datum.datum), wrap_stamp(datum.stamp));
  return datum_p;
}
}
}
}
