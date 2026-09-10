// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>

#include <viame/algorithm_framework/plugin/pluggable.h>
#include <vital/test_interface/say.h>

namespace kv = kwiver::vital;
namespace py = pybind11;

// ----------------------------------------------------------------------------
// Trampoline
class trampoline_say : public kv::say
{
public:
  using say::say;

  /// Pure virtual trampoline method
  std::string
  says() override
  {
    PYBIND11_OVERRIDE_PURE(
      std::string,
      kv::say,
      says
    );
  }
};

// ----------------------------------------------------------------------------
PYBIND11_MODULE( _interface, m )
{
  // import pluggable python type for class definition.
  py::module::import( "kwiver.vital.plugins._pluggable" );

  py::class_<
    kv::say,
    kv::pluggable,
    kv::say_sptr,
    trampoline_say
  >( m, "Say", "Test interface for outputting a simple string." )
    .def( py::init<>() )
    .def_static( "interface_name", &kv::say::interface_name )
    .def( "says", &kv::say::says )
  ;

  m.def(
    "call_says",
    []( kv::say_sptr const& inst ) -> std::string {
      py::print( "In C++ `call_says()`, about to call `inst->says()`..." );
      return inst->says();
    },
    "Tester function to get the given implementation to speak."
  );
}
