// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include "algorithm_python.txx"
#include "extract_descriptors_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void extract_descriptors(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::extract_descriptors,
               std::shared_ptr<viame::algo::extract_descriptors>,
               viame::algorithm,
               extract_descriptors_trampoline<> > instance(m,  "ExtractDescriptors");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::extract_descriptors::interface_name)
    .def("extract", &viame::algo::extract_descriptors::extract, py::doc(R"( Extract from the image a descriptor corresoponding to each feature

 \param [in]     image_data contains the image data to process
 \param [in,out] features the feature locations at which descriptors
                 are extracted (may be modified).
 \param [in]     image_mask Mask image of the same dimensions as
                            \p image_data where positive values indicate
                            regions of \p image_data to consider.
 \returns a set of feature descriptors

 \note The feature_set passed into this function may modified to
       reorder, remove, or duplicate some features to align with the
       set of descriptors detected.  If the feature_set needs to change,
       a new feature_set is created and returned by reference.)"), py::arg("image_data"), py::arg("features"), py::arg("image_mask") = py::none())
    ;
  register_algorithm< viame::algo::extract_descriptors > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
