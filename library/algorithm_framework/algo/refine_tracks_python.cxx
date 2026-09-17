// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.


#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/algo/refine_tracks.h>
#include "algorithm_python.txx"
#include "refine_tracks_trampoline_python.txx"

namespace viame::python {
namespace py = pybind11;

void refine_tracks(py::module& m)
{
  py::module::import("viame.config");
  py::module::import("viame.types");

    py::class_<viame::algo::refine_tracks,
               std::shared_ptr<viame::algo::refine_tracks>,
               viame::algorithm,
               refine_tracks_trampoline<> > instance(m,  "RefineTracks");
    
    instance
    .def(py::init<>())
    .def_static("interface_name", &viame::algo::refine_tracks::interface_name)
    .def("refine", &viame::algo::refine_tracks::refine, py::doc(R"( Refine all object tracks for the current frame.

 This method analyzes the supplied image and tracks, returning
 a refined set of tracks for the current frame.

 \param ts Timestamp for the current frame
 \param image_data The image pixels for the current frame
 \param tracks Object tracks to refine (containing states for current
 frame)
 \returns Refined object track set)"), py::arg("ts"), py::arg("image_data"), py::arg("tracks"))
    .def("finalize", &viame::algo::refine_tracks::finalize, py::doc(R"( Finalize the refiner after all frames have been processed.

 Called when the pipeline signals completion.  Implementations may
 override this to run deferred processing (e.g. video propagation
 over the full accumulated buffer).

 \returns Final refined object track set, or nullptr if no final
          output is needed.)"))
    ;
  register_algorithm< viame::algo::refine_tracks > (instance);
}

}
#undef KWIVER_PYBIND11_INCLUDE
