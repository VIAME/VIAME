// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file read_object_track_set_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<read_object_track_set> and read_object_track_set
 */

#ifndef READ_OBJECT_TRACK_SET_TRAMPOLINE_TXX
#define READ_OBJECT_TRACK_SET_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/read_object_track_set.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_rots_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::read_object_track_set > >
class algorithm_def_rots_trampoline :
      public algorithm_trampoline<algorithm_def_rots_base>
{
  public:
    using algorithm_trampoline<algorithm_def_rots_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::read_object_track_set>,
        type_name,
      );
    }
};

template< class read_object_track_set_base=
                kwiver::vital::algo::read_object_track_set >
class read_object_track_set_trampoline :
      public algorithm_def_rots_trampoline< read_object_track_set_base >
{
  public:
    using algorithm_def_rots_trampoline< read_object_track_set_base>::
              algorithm_def_rots_trampoline;
    void
    open( std::string const& filename ) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::read_object_track_set,
        open,
        filename
      );
    }

    void
    close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::read_object_track_set,
        close,
      );
    }

    bool
    read_set( kwiver::vital::object_track_set_sptr& track_set ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::read_object_track_set,
        read_set,
        track_set
      );
    }
};

}
}
}

#endif
