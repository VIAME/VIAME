// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file match_descriptor_sets_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<match_descriptor_sets> and match_descriptor_sets
 */

#ifndef MATCH_DESCRIPTOR_SETS_TRAMPOLINE_TXX
#define MATCH_DESCRIPTOR_SETS_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/match_descriptor_sets.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_mds_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::match_descriptor_sets > >
class algorithm_def_mds_trampoline :
      public algorithm_trampoline<algorithm_def_mds_base>
{
  public:
    using algorithm_trampoline<algorithm_def_mds_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::match_descriptor_sets>,
        type_name,
      );
    }
};

template< class match_descriptor_sets_base=
                kwiver::vital::algo::match_descriptor_sets >
class match_descriptor_sets_trampoline :
      public algorithm_def_mds_trampoline< match_descriptor_sets_base >
{
  public:
    using algorithm_def_mds_trampoline< match_descriptor_sets_base>::
              algorithm_def_mds_trampoline;

    void
    append_to_index( kwiver::vital::descriptor_set_sptr const tracks,
                     kwiver::vital::frame_id_t frame ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::match_descriptor_sets,
        append_to_index,
        tracks,
        frame
      );
    }

    std::vector< kwiver::vital::frame_id_t >
    query( kwiver::vital::descriptor_set_sptr const desc ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        std::vector< kwiver::vital::frame_id_t >,
        kwiver::vital::algo::match_descriptor_sets,
        query,
        desc
      );
    }

    std::vector< kwiver::vital::frame_id_t >
    query_and_append( kwiver::vital::descriptor_set_sptr const desc,
                      kwiver::vital::frame_id_t frame ) override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        std::vector< kwiver::vital::frame_id_t >,
        kwiver::vital::algo::match_descriptor_sets,
        query_and_append,
        desc,
        frame
      );
    }
};

}
}
}

#endif
