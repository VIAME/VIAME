// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file analyze_tracks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::analyze_tracks track analyzer \endlink
 */

#ifndef ANALYZE_TRACKS_TRAMPOLINE_TXX
#define ANALYZE_TRACKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/analyze_tracks.h>

#include <ostream>

namespace kwiver {
namespace vital  {
namespace python {

template< class algorithm_def_at_base=
           kwiver::vital::algorithm_def< kwiver::vital::algo::analyze_tracks > >
class algorithm_def_at_trampoline :
      public algorithm_trampoline< algorithm_def_at_base>
{
  public:
    using algorithm_trampoline< algorithm_def_at_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::analyze_tracks>,
        type_name,
      );
    }
};

template< class analyze_tracks_base=kwiver::vital::algo::analyze_tracks >
class analyze_tracks_trampoline :
      public algorithm_def_at_trampoline< analyze_tracks_base >
{
  public:
    using algorithm_def_at_trampoline< analyze_tracks_base >::
              algorithm_def_at_trampoline;

    void print_info( kwiver::vital::track_set_sptr track_set,
                     std::ostream& stream = std::cout ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::analyze_tracks,
        print_info,
        track_set,
        stream
      );
    }
};
}
}
}
#endif
