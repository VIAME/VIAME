// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file write_object_track_set_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<write_object_track_set> and write_object_track_set
 */

#ifndef WRITE_OBJECT_TRACK_SET_TRAMPOLINE_TXX
#define WRITE_OBJECT_TRACK_SET_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/write_object_track_set.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_wots_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::write_object_track_set > >
class algorithm_def_wots_trampoline :
      public algorithm_trampoline<algorithm_def_wots_base>
{
  public:
    using algorithm_trampoline<algorithm_def_wots_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::write_object_track_set>,
        type_name,
      );
    }
};

template< class write_object_track_set_base=
                kwiver::vital::algo::write_object_track_set >
class write_object_track_set_trampoline :
      public algorithm_def_wots_trampoline< write_object_track_set_base >
{
  public:
    using algorithm_def_wots_trampoline< write_object_track_set_base>::
              algorithm_def_wots_trampoline;

    void open( std::string const& filename ) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::write_object_track_set,
        open,
        filename
      );
    }

    void close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::write_object_track_set,
        close,
      );
    }

    void
    write_set(const kwiver::vital::object_track_set_sptr& set,
              kwiver::vital::timestamp const& ts = {},
              std::string const& frame_identifier = {}) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::write_object_track_set,
        write_set,
        set,
        ts,
        frame_identifier
      );
    }
};

}
}
}

#endif
