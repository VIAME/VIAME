// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief Hand-written trampoline for extract_descriptors
///
/// Replaces the generated one, which cannot carry the replaced feature set
/// back from python. See trampolines/README.md.

#ifndef EXTRACT_DESCRIPTORS_TRAMPOLINE_TXX
#define EXTRACT_DESCRIPTORS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <viame/core_types/casters.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "algorithm_trampoline_python.txx"
#include "out_parameter_python.txx"
#include <viame/algorithm_framework/algo/extract_descriptors.h>

namespace viame::python {

template< class extract_descriptors_base =
            viame::algo::extract_descriptors >
class extract_descriptors_trampoline
  : public algorithm_trampoline< extract_descriptors_base >
{
public:
  using algorithm_trampoline< extract_descriptors_base >::algorithm_trampoline;

  viame::descriptor_set_sptr
  extract(
    ::viame::image_container_sptr image_data,
    ::viame::feature_set_sptr& features,
    ::viame::image_container_sptr image_mask ) const override
  {
    pybind11::gil_scoped_acquire gil;
    pybind11::function overload =
      pybind11::get_override(
        static_cast< viame::algo::extract_descriptors const* >( this ),
        "extract" );

    if( !overload )
    {
      pybind11::pybind11_fail(
        "Tried to call pure virtual function "
        "\"extract_descriptors::extract\"" );
    }

    return unpack_out_parameters< viame::descriptor_set_sptr >(
      overload( image_data, features, image_mask ),
      "extract_descriptors.extract", features );
  }
};

} // namespace viame::python

#undef KWIVER_PYBIND11_INCLUDE
#endif
