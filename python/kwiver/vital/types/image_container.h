// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_VITAL_PYTHON_IMAGE_CONTAINER_H_
#define KWIVER_VITAL_PYTHON_IMAGE_CONTAINER_H_

#include <pybind11/pybind11.h>
#include <vital/types/image_container.h>
#include <python/kwiver/vital/types/image.h>

namespace py = pybind11;

typedef kwiver::vital::image_container image_cont_t;
typedef kwiver::vital::simple_image_container s_image_cont_t;

namespace kwiver {
namespace vital  {
namespace python {
namespace image_container {
void image_container(py::module &m);

// We need to return a shared pointer--otherwise, pybind11 may lose the subtype
std::shared_ptr<s_image_cont_t>new_cont(kwiver::vital::image &img);

// We need to do a deep copy instead of just calling get_image, so we can ref track in python
kwiver::vital::image get_image(std::shared_ptr<image_cont_t> self);

template <typename T>
s_image_cont_t new_image_container_from_numpy(py::array_t<T> array)
{
  kwiver::vital::image img = kwiver::vital::python::image::new_image_from_numpy(array);
  return s_image_cont_t(img);
}

} } } }

#endif
