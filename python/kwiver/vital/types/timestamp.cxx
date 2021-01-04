// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/timestamp.h>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/embed.h>

namespace py = pybind11;

using ts = kwiver::vital::timestamp;

PYBIND11_MODULE(timestamp, m)
{
  py::class_<ts, std::shared_ptr<ts> > (m, "Timestamp", R"(
     timestamp for video image.

    Example:
        >>> from kwiver.vital.types import *
        >>> ts = Timestamp(13245, 10)
        >>> print(str(ts))
        ts(f: 10, t: 13245 (Thu Jan  1 00:00:00 1970), d: 0)
    )")

    .def(py::init<>())
    .def(py::init<kwiver::vital::time_usec_t, kwiver::vital::frame_id_t>())

    .def("is_valid", &ts::is_valid)
    .def("has_valid_time", &ts::has_valid_time)
    .def("has_valid_frame", &ts::has_valid_frame)
    .def("get_time_usec", &ts::get_time_usec)
    .def("get_time_seconds", &ts::get_time_seconds)
    .def("get_frame", &ts::get_frame)
    .def("get_time_domain_index", &ts::get_time_domain_index)

    .def("set_invalid", &ts::set_invalid)
    .def("set_time_usec", &ts::set_time_usec, py::arg("time"))
    .def("set_time_seconds", &ts::set_time_seconds, py::arg("time"))
    .def("set_frame", &ts::set_frame, py::arg("frame"))
    .def("set_time_domain_index", &ts::set_time_domain_index, py::arg("domain"))

    .def("__nice__", [](ts& self) -> std::string {
        auto locals = py::dict(py::arg("self")=self);
        py::exec(R"(
        retval = '{}, {}'.format(self.get_time_seconds(), self.get_frame())
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

    .def("__str__", &kwiver::vital::timestamp::pretty_print)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self <= py::self)
    .def(py::self >= py::self)
    .def(py::self < py::self)
    .def(py::self > py::self)
    ;
}
