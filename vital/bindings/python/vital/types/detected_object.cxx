/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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

#include <vital/types/detected_object.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

namespace py = pybind11;

typedef kwiver::vital::detected_object det_obj;

// We want to be able to add a mask in the python constructor
// so we need a pass-through cstor
std::shared_ptr<det_obj>
new_detected_object(kwiver::vital::bounding_box<double> bbox,
                    double conf,
                    kwiver::vital::detected_object_type_sptr type,
                    kwiver::vital::image_container_sptr mask)
{
  std::shared_ptr<det_obj> new_obj(new det_obj(bbox, conf, type));

  if(mask)
  {
    new_obj->set_mask(mask);
  }

  return new_obj;
}

PYBIND11_MODULE(detected_object, m)
{
  /*
   *
    
    Developer:
        python -c "import vital.types; help(vital.types.DetectedObject)"
        python -m xdoctest vital.types DetectedObject --xdoc-dynamic

   *
   */

  py::class_<det_obj, std::shared_ptr<det_obj>>(m, "DetectedObject", R"(
    Represents a detected object within an image

    Example:
        >>> from vital.types import *
        >>> from PIL import Image as PILImage
        >>> from vital.util import VitalPIL
        >>> import numpy as np
        >>> bbox = BoundingBox(0, 10, 100, 50)
        >>> # Construct an object without a mask
        >>> dobj1 = DetectedObject(bbox, 0.2)
        >>> assert dobj1.mask is None
        >>> # Construct an object with a mask
        >>> pil_img = PILImage.fromarray(np.zeros((10, 10), dtype=np.uint8))
        >>> vital_img = VitalPIL.from_pil(pil_img)
        >>> mask = ImageContainer(vital_img)
        >>> self = DetectedObject(bbox, 1.0, mask=mask)
        >>> assert self.mask is mask
        >>> print(self)
        <DetectedObject(conf=1.0)>
    )")
  .def(py::init(&new_detected_object),
    py::arg("bbox"), py::arg("confidence")=1.0,
    py::arg("classifications")=kwiver::vital::detected_object_type_sptr(),
    py::arg("mask")=kwiver::vital::image_container_sptr(), py::doc(R"(
      Args:
          bbox: coarse localization of the object in image coordinates
          confidence: confidence in this detection (default=1.0)
          classifications: optional object classification (default=None)
    ")"))
  .def(py::init<kwiver::vital::bounding_box<double>, double, kwiver::vital::detected_object_type_sptr>(),
    py::arg("bbox"), py::arg("confidence")=1.0, py::arg("classifications")=kwiver::vital::detected_object_type_sptr())
  .def("bounding_box", &det_obj::bounding_box)
  .def("set_bounding_box", &det_obj::set_bounding_box,
    py::arg("bbox"))
  .def("confidence", &det_obj::confidence)
  .def("set_confidence", &det_obj::set_confidence,
    py::arg("d"))
  .def("descriptor", &det_obj::descriptor)
  .def("set_descriptor", &det_obj::set_descriptor,
		py::arg("descriptor"))
  .def("type", &det_obj::type)
  .def("set_type", &det_obj::set_type,
    py::arg("c"))
  .def_property("mask", &det_obj::mask, &det_obj::set_mask)
  .def("__nice__", [](det_obj& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        retval = 'conf={}'.format(self.confidence())
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  .def("__repr__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  .def("__str__", [](py::object& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        classname = self.__class__.__name__
        devnice = self.__nice__()
        retval = '<%s(%s)>' % (classname, devnice)
    )", py::globals(), locals);
    return locals["retval"].cast<std::string>();
    })
  ;
}
