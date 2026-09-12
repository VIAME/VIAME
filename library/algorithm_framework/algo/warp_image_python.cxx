// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/warp_image.h>
#include "algorithm_python.txx"
#include "warp_image_trampoline_python.txx"

namespace kwiver::vital::python {
namespace py = pybind11;

void warp_image(py::module& m)
{
  py::module::import("kwiver.vital.config");
  py::module::import("kwiver.vital.types");

    py::class_<kwiver::vital::algo::warp_image,
               std::shared_ptr<kwiver::vital::algo::warp_image>,
               kwiver::vital::algorithm,
               warp_image_trampoline<> > instance(m,  "WarpImage");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &kwiver::vital::algo::warp_image::interface_name)
    .def("warp", &kwiver::vital::algo::warp_image::warp, py::doc(R"( Warp \p src_image onto \p dst_image.

 \param src_image Source image to draw pixel values from.
 \param dst_image Destination image to draw pixel values to.
 \param homography
   Homography mapping \p src_image to \p dst_image, in pixels.
 \param alpha_mask
   Optional single-channel image indicating the opacity of \p src_image.

 \return
   Result after warping. This may be \p dst_image or a new image object.
   Implementations are encouraged to perform the operation in-place
   (returning the modified \p dst_image ) if possible.)"), py::arg("src_image"), py::arg("dst_image"), py::arg("homography"), py::arg("alpha_mask"))
    ;
  register_algorithm< kwiver::vital::algo::warp_image > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
