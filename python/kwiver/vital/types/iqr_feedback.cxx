// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/iqr_feedback.h>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

PYBIND11_MODULE(iqr_feedback, m)
{
  py::class_<kv::iqr_feedback, std::shared_ptr<kv::iqr_feedback>>(m, "IQRFeedback")
  .def(py::init<>())
  .def_property("query_id", &kv::iqr_feedback::query_id, &kv::iqr_feedback::set_query_id)
  .def_property("positive_ids", &kv::iqr_feedback::positive_ids, &kv::iqr_feedback::set_positive_ids)
  .def_property("negative_ids", &kv::iqr_feedback::negative_ids, &kv::iqr_feedback::set_negative_ids)
  ;
}
