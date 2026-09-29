// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef MATCH_DESCRIPTOR_SETS_TRAMPOLINE_TXX
#define MATCH_DESCRIPTOR_SETS_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/match_descriptor_sets.h>

namespace viame::python {

template< class match_descriptor_sets_base = viame::algo::match_descriptor_sets >
class match_descriptor_sets_trampoline
    : public algorithm_trampoline< match_descriptor_sets_base >
{
  public:
    using algorithm_trampoline< match_descriptor_sets_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  void
  append_to_index(::viame::descriptor_set_sptr const desc, ::viame::frame_id_t frame) override
  {
    PYBIND11_OVERLOAD_PURE(
      void,
      viame::algo::match_descriptor_sets,
      append_to_index,
      desc, frame
      );
  }

  std::vector< ::viame::frame_id_t >
  query(::viame::descriptor_set_sptr const desc) override
  {
    PYBIND11_OVERLOAD_PURE(
      std::vector< ::viame::frame_id_t >,
      viame::algo::match_descriptor_sets,
      query,
      desc
      );
  }

  std::vector< ::viame::frame_id_t >
  query_and_append(::viame::descriptor_set_sptr const desc, ::viame::frame_id_t frame) override
  {
    PYBIND11_OVERLOAD(
      std::vector< ::viame::frame_id_t >,
      viame::algo::match_descriptor_sets,
      query_and_append,
      desc, frame
      );
  }
}; // class
} // namespace viame::python
#undef KWIVER_PYBIND11_INCLUDE
#endif
