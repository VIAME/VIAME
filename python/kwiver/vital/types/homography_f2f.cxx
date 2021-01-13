// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#include <vital/types/homography_f2f.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <vital/types/homography.h>
#include <vital/vital_types.h>

#include <Eigen/Core>
#include <stdexcept>
#include <sstream>

namespace kv = kwiver::vital;
namespace py = pybind11;

using f2f_homography = kv::f2f_homography;
using float_mat_t =    Eigen::Matrix<float,  3, 3 >;
using double_mat_t =   Eigen::Matrix<double, 3, 3 >;

PYBIND11_MODULE(homography_f2f, m)
{
  py::class_<f2f_homography, std::shared_ptr<f2f_homography>>(m, "F2FHomography")
    .def(py::init<kv::frame_id_t>())
    // These are the copy constructors from templated eigen matrices
    .def_static("from_floats",
      [] (float_mat_t const& mat, kv::frame_id_t const from_id, kv::frame_id_t const to_id)
      {
        return f2f_homography(mat, from_id, to_id);
      })
    .def_static("from_doubles",
      [] (double_mat_t const& mat, kv::frame_id_t const from_id, kv::frame_id_t const to_id)
      {
        return f2f_homography(mat, from_id, to_id);
      })
    .def(py::init<kv::homography_sptr const&, kv::frame_id_t const, kv::frame_id_t const>())
    .def(py::init<f2f_homography const&>())
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
    .def("__str__", [] (f2f_homography const& self)
    {
      std::stringstream str;
      str << self;
      return str.str();
    })
    ;
}
