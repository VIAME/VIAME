// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/detected_object_set.h>
#include <python/kwiver/vital/util/pybind11.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

namespace py = pybind11;

typedef kwiver::vital::detected_object_set det_obj_set;
using detected_object_sptr = std::shared_ptr< kwiver::vital::detected_object >;
using detected_object_scptr = std::shared_ptr< kwiver::vital::detected_object const >;

class det_obj_set_trampoline : public kwiver::vital::detected_object_set{

  using det_obj_set::det_obj_set;
  size_t size() const override;
  bool empty() const override;
  detected_object_sptr at(size_t pos) override;
  const detected_object_sptr at(size_t pos) const override;
};

size_t det_obj_set_trampoline::size() const
{
  VITAL_PYBIND11_OVERLOAD(
    size_t,
    det_obj_set,
    size,
  );
}

bool det_obj_set_trampoline::empty() const
{
  VITAL_PYBIND11_OVERLOAD(
    bool,
    det_obj_set,
    empty,
  );
}

const detected_object_sptr det_obj_set_trampoline::at(size_t pos) const
{
  VITAL_PYBIND11_OVERLOAD(
    detected_object_sptr,
    det_obj_set,
    at,
    pos
  );
}

detected_object_sptr det_obj_set_trampoline::at(size_t pos)
{
  VITAL_PYBIND11_OVERLOAD(
    detected_object_sptr,
    det_obj_set,
    at,
    pos
  );
}


PYBIND11_MODULE(detected_object_set, m)
{
  /*
   *

    Developer:
        python -c "import kwiver.vital.types; help(kwiver.vital.types.DetectedObjectSet)"
        python -m xdoctest kwiver.vital.types DetectedObjectSet --xdoc-dynamic

   *
   */
  py::class_<det_obj_set, std::shared_ptr<det_obj_set>, det_obj_set_trampoline>(m, "DetectedObjectSet", R"(
      Collection holding a multiple detected objects

      Example:
          >>> from kwiver.vital.types import *
          >>> bbox = BoundingBox(0, 10, 100, 50)
          >>> dobj1 = DetectedObject(bbox, 0.2)
          >>> dobj2 = DetectedObject(bbox, 0.5)
          >>> dobj3 = DetectedObject(bbox, 0.4)
          >>> self = DetectedObjectSet()
          >>> self.add(dobj1)
          >>> self.add(dobj2)
          >>> self.add(dobj3)
          >>> self.add(dobj3)
          >>> assert len(self) == 4
          >>> assert list(self) == [dobj1, dobj2, dobj3, dobj3]
          >>> assert list(self) != [dobj1, dobj1, dobj1, dobj1]
          >>> print(self)
          <DetectedObjectSet(size=4)>
    )")
  .def(py::init<>())
  .def(py::init<std::vector<std::shared_ptr<kwiver::vital::detected_object>>>())
  .def("add", [](det_obj_set &self, py::object object)
    {
      try
      {
        auto det_obj = object.cast<std::shared_ptr<kwiver::vital::detected_object>>();
        return self.add(det_obj);
      }
      catch(...){};
      auto det_obj = object.cast<std::shared_ptr<det_obj_set>>();
      return self.add(det_obj);
    })
  .def("clone", &det_obj_set::clone)
  .def("at", (const detected_object_sptr (det_obj_set::*)(size_t) const) &det_obj_set::at)
  .def("at", (detected_object_sptr (det_obj_set::*)(size_t)) &det_obj_set::at)
  .def("empty", &det_obj_set::empty)
  .def("size", &det_obj_set::size)
  .def("__len__", &det_obj_set::size)
  .def("select", [](det_obj_set &self, double threshold, py::object class_name)
    {
      if(class_name.is(py::none()))
      {
        return self.select(threshold);
      }
      return self.select(class_name.cast<std::string>(), threshold);
    },
    py::arg("threshold")=kwiver::vital::detected_object_type::INVALID_SCORE, py::arg("class_name")=py::none())
  .def("__getitem__", [](det_obj_set &self, size_t idx)
    {
      return self.at(idx);
    })

  .def("__iter__", [](det_obj_set &self) { return py::make_iterator(self.cbegin(), self.cend()); },
          py::keep_alive<0, 1>())

  .def("__nice__", [](det_obj_set& self) -> std::string {
    auto locals = py::dict(py::arg("self")=self);
    py::exec(R"(
        retval = 'size={}'.format(len(self))
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
