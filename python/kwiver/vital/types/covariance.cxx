// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "covariance_class.cxx"

namespace py = pybind11;

using namespace kwiver::vital::python;
PYBIND11_MODULE(covariance, m)
{
  py::class_<PyCovarianceBase, std::shared_ptr<PyCovarianceBase> >(m, "Covariance")
  .def_static("new_covar", &PyCovarianceBase::covar_from_scalar, // need to use a factory func instead of constructor
       py::arg("N")=2, py::arg("c_type")='d', py::arg("init")=py::none())
  .def_static("from_matrix", &PyCovarianceBase::covar_from_matrix,
       py::arg("N")=2, py::arg("c_type")='d', py::arg("init")=py::none())
  .def("to_matrix", &PyCovarianceBase::to_matrix)
  .def("__setitem__", [](PyCovarianceBase &self, py::tuple idx, py::object value)
                      {
                        self.set_item(idx[0].cast<int>(), idx[1].cast<int>(), value);
                      })
  .def("__getitem__", [](PyCovarianceBase &self, py::tuple idx)
                      {
                        return self.get_item(idx[0].cast<int>(), idx[1].cast<int>());
                      })
   ;

  // it's nice to be able to directly use the subclasses
  py::class_<PyCovariance2d, std::shared_ptr<PyCovariance2d>, PyCovarianceBase>(m, "Covar2d");
  py::class_<PyCovariance2f, std::shared_ptr<PyCovariance2f>, PyCovarianceBase>(m, "Covar2f");
  py::class_<PyCovariance3d, std::shared_ptr<PyCovariance3d>, PyCovarianceBase>(m, "Covar3d");
  py::class_<PyCovariance3f, std::shared_ptr<PyCovariance3f>, PyCovarianceBase>(m, "Covar3f");

}
