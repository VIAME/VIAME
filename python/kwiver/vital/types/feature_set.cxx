// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/types/feature_set.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

class feature_set_trampoline
  : public kv::feature_set
{
public:
  using feature_set::feature_set;
  size_t size() const override;
  std::vector< kv::feature_sptr > features() const override;
};

PYBIND11_MODULE( feature_set, m )
{
  py::class_< kv::feature_set,
              std::shared_ptr< kv::feature_set >,
              feature_set_trampoline >( m, "FeatureSet" )
  .def( py::init<>() )
  .def( "size", &kv::feature_set::size )
  .def( "features", &kv::feature_set::features )
  ;

  py::class_< kv::simple_feature_set,
              std::shared_ptr< kv::simple_feature_set >,
              kv::feature_set >( m, "SimpleFeatureSet" )
  .def( py::init<>() )
  .def( py::init< const std::vector< kv::feature_sptr >& >() )
  .def( "size", &kv::simple_feature_set::size )
  .def( "features", &kv::simple_feature_set::features )
  ;
}

size_t
feature_set_trampoline
::size() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    size_t,
    kv::feature_set,
    size,
  );
}

std::vector< kv::feature_sptr >
feature_set_trampoline
::features() const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    std::vector< kv::feature_sptr >,
    kv::feature_set,
    features,
  );
}
