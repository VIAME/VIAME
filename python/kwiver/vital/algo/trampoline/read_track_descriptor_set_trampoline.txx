// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file read_track_descriptor_set_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<read_track_descriptor_set> and read_track_descriptor_set
 */

#ifndef READ_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX
#define READ_TRACK_DESCRIPTOR_SET_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/read_track_descriptor_set.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_rtds_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::read_track_descriptor_set > >
class algorithm_def_rtds_trampoline :
      public algorithm_trampoline<algorithm_def_rtds_base>
{
  public:
    using algorithm_trampoline<algorithm_def_rtds_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::read_track_descriptor_set>,
        type_name,
      );
    }
};

template< class read_track_descriptor_set_base=
                kwiver::vital::algo::read_track_descriptor_set >
class read_track_descriptor_set_trampoline :
      public algorithm_def_rtds_trampoline< read_track_descriptor_set_base >
{
  public:
    using algorithm_def_rtds_trampoline< read_track_descriptor_set_base>::
              algorithm_def_rtds_trampoline;
    void
    open( std::string const& filename ) override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::read_track_descriptor_set,
        open,
        filename
      );
    }

    void
    close() override
    {
      VITAL_PYBIND11_OVERLOAD(
        void,
        kwiver::vital::algo::read_track_descriptor_set,
        close,
      );
    }

    bool
    read_set( kwiver::vital::track_descriptor_set_sptr& descriptor_set ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        bool,
        kwiver::vital::algo::read_track_descriptor_set,
        read_set,
        descriptor_set
      );
    }
};

}
}
}

#endif
