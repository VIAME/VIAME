// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/vital/algo/trampoline/image_io_trampoline.txx>
#include <python/kwiver/vital/algo/image_io.h>

#include <pybind11/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
namespace py = pybind11;

class py_image_io : public kwiver::vital::algo::image_io
{
  public:
    using kwiver::vital::algo::image_io::set_capability;
};

void image_io(py::module &m)
{
  py::class_< kwiver::vital::algo::image_io,
              std::shared_ptr<kwiver::vital::algo::image_io>,
              kwiver::vital::algorithm_def<kwiver::vital::algo::image_io>,
              image_io_trampoline<> >(m, "ImageIO")
    .def(py::init())
    .def_static("static_type_name", &kwiver::vital::algo::image_io::static_type_name)
    .def("load", &kwiver::vital::algo::image_io::load)
    .def("load_metadata", &kwiver::vital::algo::image_io::load_metadata)
    .def("save", &kwiver::vital::algo::image_io::save)
    .def("get_implementation_capabilities",
             &kwiver::vital::algo::image_io::get_implementation_capabilities)
    .def_readonly_static("HAS_TIME",
                          &kwiver::vital::algo::image_io::HAS_TIME)
    .def("set_implementation_capabilities", &py_image_io::set_capability);
}
}
}
}
