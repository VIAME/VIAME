// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/camera.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py = pybind11;
namespace kv = kwiver::vital;
namespace kwiver {
namespace vital  {
namespace python {

class camera_trampoline
  : public kv::camera
{
  using kv::camera::camera;

  kv::camera_sptr clone() const override;
  kv::vector_2d project( const kv::vector_3d& pt ) const override;
  unsigned int image_width() const override;
  unsigned int image_height() const override;
};
}
}
}
using namespace kwiver::vital::python;
PYBIND11_MODULE( camera, m )
{
  py::class_< kv::camera,
              std::shared_ptr< kv::camera >,
              camera_trampoline>( m, "Camera" )
  .def( py::init<>() )
  .def( "project",      &kv::camera::project )
  .def( "image_width",  &kv::camera::image_width )
  .def( "image_height", &kv::camera::image_height )
  ;
}
// We are excluding clone in the base's binding code to follow the pattern
// described in this pybind issue:
// https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
// Subclasses will still be able to override it, however.
// Pybind automatically downcasts pointers returned by clone() to the lowest possible
// subtype, but under certain circumstances, the returned pointer can get sliced.
// The above link has a solution to this issue. The trampoline's implementation of
// clone() also was modified to follow this pattern


kv::camera_sptr
camera_trampoline
::clone() const
{
  auto self = py::cast(this);
  auto cloned = self.attr("clone")();

  auto keep_python_state_alive = std::make_shared<py::object>(cloned);
  auto ptr = cloned.cast<camera_trampoline*>();

  // aliasing shared_ptr: points to `camera_trampoline* ptr` but refcounts the Python object
  return std::shared_ptr<kv::camera>(keep_python_state_alive, ptr);
}

kv::vector_2d
camera_trampoline
::project( const kv::vector_3d& pt ) const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_2d,
    kv::camera,
    project,
    pt
  );
}

unsigned int
camera_trampoline
::image_width() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    unsigned int,
    kv::camera,
    image_width,
  );
}

unsigned int
camera_trampoline
::image_height() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    unsigned int,
    kv::camera,
    image_height,
  );
}
