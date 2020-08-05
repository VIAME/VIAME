/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "covariance_class.cxx"

namespace py = pybind11;

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
