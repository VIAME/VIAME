// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/category_hierarchy.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace py = pybind11;
namespace kv = kwiver::vital;

using label_t = kv::category_hierarchy::label_t;
using label_id_t = kv::category_hierarchy::label_id_t;
using label_vec_t = kv::category_hierarchy::label_vec_t;
using label_id_vec_t = kv::category_hierarchy::label_id_vec_t;

PYBIND11_MODULE( category_hierarchy, m )
{
  py::class_< kv::category_hierarchy,
              std::shared_ptr< kv::category_hierarchy > >( m, "CategoryHierarchy" )
  .def(py::init<>())
  .def(py::init< std::string >())
  .def(py::init< const label_vec_t&, const label_vec_t&, const label_id_vec_t& >(),
                 py::arg( "class_names" ), py::arg( "parent_names" ) = label_vec_t(),
                 py::arg("ids") = label_id_vec_t() )
  .def( "add_class", &kv::category_hierarchy::add_class,
        py::arg( "class_name" ), py::arg( "parent_name" ) = label_t(""),
        py::arg("id") = label_id_t(-1) )
  .def( "has_class_name", &kv::category_hierarchy::has_class_name )
  .def( "get_class_name", &kv::category_hierarchy::get_class_name )
  .def( "get_class_id", &kv::category_hierarchy::get_class_id )
  .def( "get_class_parents", &kv::category_hierarchy::get_class_parents )
  .def( "add_relationship", &kv::category_hierarchy::add_relationship )
  .def( "add_synonym", &kv::category_hierarchy::add_synonym )
  .def( "all_class_names", &kv::category_hierarchy::all_class_names )
  .def( "child_class_names", &kv::category_hierarchy::child_class_names )
  .def( "size", &kv::category_hierarchy::size )
  .def( "load_from_file", &kv::category_hierarchy::load_from_file )
  ;
}
