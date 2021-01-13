/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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
