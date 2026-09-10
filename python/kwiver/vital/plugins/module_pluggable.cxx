// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/plugin/pluggable.h>

namespace kv = kwiver::vital;
namespace py = pybind11;

PYBIND11_MODULE( _pluggable, m )
{
  py::class_< kv::pluggable, kv::pluggable_sptr >( m, "Pluggable" )
  // provide NotImplementedError raising class methods for expected
  // static/class methods that implementations are expected to provide.
  // TODO: See above, pair with motivating use-case/unit-test.
  // TODO: Would love to implement classmethods defining default
  //       implementations...
  //       Try? https://github.com/pybind/pybind11/issues/1693
  ;
}
