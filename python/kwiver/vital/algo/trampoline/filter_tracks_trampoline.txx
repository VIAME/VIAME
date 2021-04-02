// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file filter_tracks_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::filter_tracks \endlink
 */

#ifndef FILTER_TRACKS_TRAMPOLINE_TXX
#define FILTER_TRACKS_TRAMPOLINE_TXX

#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/filter_tracks.h>

#include <python/kwiver/vital/util/pybind11.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_ft_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::filter_tracks > >
class algorithm_def_ft_trampoline :
      public algorithm_trampoline< algorithm_def_ft_base >
{
  public:
    using algorithm_trampoline< algorithm_def_ft_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def< kwiver::vital::algo::filter_tracks >,
        type_name,
      );
    }
};

template< class filter_tracks_base =
                  kwiver::vital::algo::filter_tracks >
class filter_tracks_trampoline :
      public algorithm_def_ft_trampoline< filter_tracks_base >
{
  public:
    using algorithm_def_ft_trampoline< filter_tracks_base >::
              algorithm_def_ft_trampoline;

    kwiver::vital::track_set_sptr
    filter( kwiver::vital::track_set_sptr input ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::track_set_sptr,
        kwiver::vital::algo::filter_tracks,
        filter,
        input
      );
    }
};

}
}
}

#endif
