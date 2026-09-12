// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef PERFORM_TEXT_QUERY_TRAMPOLINE_TXX
#define PERFORM_TEXT_QUERY_TRAMPOLINE_TXX

#define KWIVER_PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include "algorithm_trampoline_python.txx"
#include <viame/algorithm_framework/algo/perform_text_query.h>

namespace kwiver::vital::python {

template< class perform_text_query_base = kwiver::vital::algo::perform_text_query >
class perform_text_query_trampoline
    : public algorithm_trampoline< perform_text_query_base >
{
  public:
    using algorithm_trampoline< perform_text_query_base >::algorithm_trampoline;

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  std::vector<std::shared_ptr<kwiver::vital::object_track_set> >
  perform_query(::std::string const & text_query, ::std::vector<std::shared_ptr<kwiver::vital::image_container> > const & images, ::std::vector<kwiver::vital::timestamp> const & timestamps, ::std::vector<std::shared_ptr<kwiver::vital::object_track_set> > const & input_tracks) const override
  {
    PYBIND11_OVERLOAD_PURE(
      std::vector<std::shared_ptr<kwiver::vital::object_track_set> >,
      kwiver::vital::algo::perform_text_query,
      perform_query,
      text_query, images, timestamps, input_tracks
      );
  }
}; // class
} // namespace
#undef KWIVER_PYBIND11_INCLUDE
#endif
