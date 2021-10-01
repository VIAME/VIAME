// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file interpolate_track_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<interpolate_track> and interpolate_track
 */

#ifndef INTERPOLATE_TRACK_TRAMPOLINE_TXX
#define INTERPOLATE_TRACK_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/interpolate_track.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_it_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::interpolate_track > >
class algorithm_def_it_trampoline :
      public algorithm_trampoline<algorithm_def_it_base>
{
  public:
    using algorithm_trampoline<algorithm_def_it_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::interpolate_track>,
        type_name,
      );
    }
};

template< class interpolate_track_base=
                kwiver::vital::algo::interpolate_track >
class interpolate_track_trampoline :
      public algorithm_def_it_trampoline< interpolate_track_base >
{
  public:
    using algorithm_def_it_trampoline< interpolate_track_base>::
              algorithm_def_it_trampoline;

    kwiver::vital::track_sptr
    interpolate( kwiver::vital::track_sptr init_states ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::track_sptr,
        kwiver::vital::algo::interpolate_track,
        interpolate,
        init_states
      );
    }
};

}
}
}

#endif
