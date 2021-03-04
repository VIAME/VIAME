// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/homography.h>

#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py=pybind11;
namespace kwiver {
namespace vital  {
namespace python {

namespace kv=kwiver::vital;

// Easy way to automate bindings of templated classes.
// For more information, see below link
// https://stackoverflow.com/questions/47487888/pybind11-template-class-of-many-types
template< typename T >
void declare_homogaphy( py::module &m, std::string const& typestr )
{
  using Class = kv::homography_< T >;
  using matrix_t = Eigen::Matrix< T, 3, 3>;
  const std::string pyclass_name = std::string( "Homography" ) + typestr;

  py::class_< Class, std::shared_ptr< Class >, kv::homography >(m, pyclass_name.c_str())
  .def(py::init())
  .def(py::init< matrix_t const& >())
  .def_static("random", [] ()
  {
    return Class(matrix_t::Random(3, 3));
  })
  .def("matrix", (matrix_t& (Class::*) ()) &Class::get_matrix)
  .def("inverse", &Class::inverse)
  .def("map", &Class::map,
    py::arg("point"))
  .def("normalize", &Class::normalize)
  .def("__mul__", &Class::operator*)
  .def_property_readonly( "type_name", [] ( Class const& self )
  {
    return self.data_type().name()[0];
  })
  ;
}

}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(homography, m)
{
  py::module::import("kwiver.vital.types.transform_2d");
  py::class_< kv::homography,
              kv::transform_2d,
              std::shared_ptr< kv::homography > >(m, "BaseHomography");
  declare_homogaphy< float  >(m, "F");
  declare_homogaphy< double >(m, "D");
}
