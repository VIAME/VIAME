// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <stdexcept>

#include <vital/vital_types.h>
#include <vital/types/homography.h>
#include <vital/types/homography_f2f.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>

namespace kv = kwiver::vital;
namespace py = pybind11;

using f2f_homography = kv::f2f_homography;

PYBIND11_MODULE(homography_f2f, m)
{
  // This should wrap all of f2f_homography except for the (templated)
  // constructor directly from an Eigen::Matrix and the copy
  // constructor
  py::class_<f2f_homography, std::shared_ptr<f2f_homography>>(m, "F2FHomography")
    .def(py::init<kv::homography_sptr const&, kv::frame_id_t, kv::frame_id_t>())
    .def(py::init<kv::frame_id_t>())
    .def_property_readonly("homography", &f2f_homography::homography)
    .def_property_readonly("from_id", &f2f_homography::from_id)
    .def_property_readonly("to_id", &f2f_homography::to_id)
    .def("inverse", &f2f_homography::inverse)
    .def("__mul__", &f2f_homography::operator*)
    .def("get",
	 [] (f2f_homography const& self, int r, int c)
	 {
	   auto m = self.homography()->matrix();
	   if(0 <= r && r < m.rows() && 0 <= c && c < m.cols())
	   {
	     return m(r, c);
	   }
	   throw std::out_of_range("Tried to perform get() out of bounds");
	 },
	 "Convenience method that returns the underlying coefficient"
	 " at the given row and column")
    ;
}
