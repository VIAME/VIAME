// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file read_track_descriptor_set_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<write_track_descriptor_set> and write_track_descriptor_set
 */

#ifndef WRITE_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX
#define WRITE_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/write_track_descriptor_set.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_wtds_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::write_track_descriptor_set > >
class algorithm_def_wtds_trampoline :
      public algorithm_trampoline<algorithm_def_wtds_base>
{
  public:
    using algorithm_trampoline<algorithm_def_wtds_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::write_track_descriptor_set>,
        type_name,
      );
    }
};

template< class write_track_descriptor_set_base=
                kwiver::vital::algo::write_track_descriptor_set >
class write_track_descriptor_set_trampoline :
      public algorithm_def_wtds_trampoline< write_track_descriptor_set_base >
{
  public:
    using algorithm_def_wtds_trampoline< write_track_descriptor_set_base>::
              algorithm_def_wtds_trampoline;

    void open( std::string const& filename ) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::write_track_descriptor_set,
        open,
        filename
      );
    }

    void close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::write_track_descriptor_set,
        close,
      );
    }

    void
    write_set( const kwiver::vital::track_descriptor_set_sptr set ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::write_track_descriptor_set,
        write_set,
        set
      );
    }
};

}
}
}

#endif
