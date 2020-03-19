/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
