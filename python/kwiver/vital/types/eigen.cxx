// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "eigen_class.cxx"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace kwiver::vital::python;

PYBIND11_MODULE(eigen, m)
{

  py::class_< EigenArray, std::shared_ptr<EigenArray> >(m, "EigenArray")
  .def(py::init< int, int, bool, bool, char>(),
         py::arg("rows")=2, py::arg("cols")=1,
         py::arg("dynamic_rows")=false, py::arg("dynamic_cols")=false,
         py::arg("type")='d')
  .def("get_matrix", &EigenArray::getMatrix, "Access the C++ matrix by reference", py::return_value_policy::reference_internal)
  .def("copy_matrix", &EigenArray::getMatrix, "Creates a python copy of the matrix")
  .def_static("from_array", &EigenArray::fromArray,
         py::arg("data"), py::arg("type")='d')
  ;

}
