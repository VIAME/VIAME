// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file detected_object_set_input_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of algorithm_def<detected_object_set_input> and detected_object_set_input
 */

#ifndef DETECTED_OBJECT_SET_INPUT_TRAMPOLINE_TXX
#define DETECTED_OBJECT_SET_INPUT_TRAMPOLINE_TXX

#include <tuple>

#include <python/kwiver/vital/util/pybind11.h>
#include <vital/algo/detected_object_set_input.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image_container.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>

using dosi = kwiver::vital::algo::detected_object_set_input;

namespace kwiver {
namespace vital {
namespace python {

template <class algorithm_def_dosi_base=kwiver::vital::algorithm_def<dosi>>
class algorithm_def_dosi_trampoline :
      public algorithm_trampoline<algorithm_def_dosi_base>
{
  public:
    using algorithm_trampoline<algorithm_def_dosi_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<dosi>,
        type_name,
      );
    }
};

template <class detected_object_set_input_base=dosi>
class detected_object_set_input_trampoline :
      public algorithm_def_dosi_trampoline<detected_object_set_input_base>
{
  using super = algorithm_def_dosi_trampoline<detected_object_set_input_base>;

  public:
    using super::super;

    bool read_set(kwiver::vital::detected_object_set_sptr& set, std::string& image_path) override
    {
      kwiver::vital::python::gil_scoped_acquire gil;
      pybind11::function overload = pybind11::get_overload(static_cast<dosi const*>(this), "read_set");
      if (overload) {
	auto o = overload();
	if (pybind11::isinstance<pybind11::none>(o)) {
	  return false;
	}
	std::tie(set, image_path) = o.cast<std::tuple<kwiver::vital::detected_object_set_sptr, std::string>>();
	return true;
      } else {
	pybind11::pybind11_fail("Tried to call pure virtual function \"dosi::read_set\"");
      }
    }

    void open(std::string const& filename) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        dosi,
        open,
	filename
      );
    }

    void close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        dosi,
        close,
      );
    }
};

}
}
}
#endif
