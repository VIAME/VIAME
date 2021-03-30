// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/database_query.h>

#include <pybind11/pybind11.h>
#include <memory>
#include <pybind11/stl.h>

namespace py=pybind11;
namespace kv=kwiver::vital;

// Note that this is unlike database_query::descriptors().
// This returns a copy of the track_descriptor_set, while
// database_query::descriptors() returns a pointer to
// the internal track_descriptor_set. The returned set cannot
//  be used to modify by reference
kv::track_descriptor_set
database_query_get_descriptors(const kv::database_query& self)
{
  kv::track_descriptor_set_sptr descs_sptr = self.descriptors();
  if (!descs_sptr)
  {
    return kv::track_descriptor_set();
  }
  return *descs_sptr;
}

// Pybind wont allow track_descriptor_sets to be passed around with smart_pointers
// We'll make a smart_ptr to a copy of the track_descriptor_set passed in
void
database_query_set_descriptors(kv::database_query& self, const kv::track_descriptor_set& tdset)
{
  // Note that we're making a copy of a vector of pointers.
  // Elements of the vector can still be modified by reference on the python side
  self.set_descriptors(kv::track_descriptor_set_sptr(new kv::track_descriptor_set(tdset)));
}

PYBIND11_MODULE(database_query, m)
{
  py::class_<kv::database_query, std::shared_ptr<kv::database_query>>(m, "DatabaseQuery")
  .def(py::init<>())
  .def_property("id", &kv::database_query::id, &kv::database_query::set_id)
  .def_property("type", &kv::database_query::type, &kv::database_query::set_type)
  .def_property("temporal_filter", &kv::database_query::temporal_filter, &kv::database_query::set_temporal_filter)
  .def_property("spatial_filter", &kv::database_query::spatial_filter, &kv::database_query::set_spatial_filter)
  .def_property("spatial_region", &kv::database_query::spatial_region, &kv::database_query::set_spatial_region)
  .def_property("stream_filter", &kv::database_query::stream_filter, &kv::database_query::set_stream_filter)
  .def_property("descriptors", &database_query_get_descriptors, &database_query_set_descriptors)
  .def_property("threshold", &kv::database_query::threshold, &kv::database_query::set_threshold)
  .def("temporal_lower_bound", &kv::database_query::temporal_lower_bound)
  .def("temporal_upper_bound", &kv::database_query::temporal_upper_bound)
  .def("set_temporal_bounds", &kv::database_query::set_temporal_bounds)
  ;

  py::enum_<kv::query_filter>(m, "query_filter")
  .value("IGNORE_FILTER", kv::query_filter::IGNORE_FILTER)
  .value("CONTAINS_WHOLLY", kv::query_filter::CONTAINS_WHOLLY)
  .value("CONTAINS_PARTLY", kv::query_filter::CONTAINS_PARTLY)
  .value("INTERSECTS", kv::query_filter::INTERSECTS)
  .value("INTERSECTS_INBOUND", kv::query_filter::INTERSECTS_INBOUND)
  .value("INTERSECTS_OUTBOUND", kv::query_filter::INTERSECTS_OUTBOUND)
  .value("DOES_NOT_CONTAIN", kv::query_filter::DOES_NOT_CONTAIN)
  ;

  py::enum_<kv::database_query::query_type>(m, "query_type")
  .value("SIMILARITY", kv::database_query::query_type::SIMILARITY)
  .value("RETRIEVAL", kv::database_query::query_type::RETRIEVAL)
  ;
}
