// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file match_features_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<match_features> and match_features
 */

#ifndef MATCH_FEATURES_TRAMPOLINE_TXX
#define MATCH_FEATURES_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/match_features.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_mf_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::match_features > >
class algorithm_def_mf_trampoline :
      public algorithm_trampoline<algorithm_def_mf_base>
{
  public:
    using algorithm_trampoline<algorithm_def_mf_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::match_features>,
        type_name,
      );
    }
};

template< class match_features_base=
                kwiver::vital::algo::match_features >
class match_features_trampoline :
      public algorithm_def_mf_trampoline< match_features_base >
{
  public:
    using algorithm_def_mf_trampoline< match_features_base>::
              algorithm_def_mf_trampoline;

    kwiver::vital::match_set_sptr
    match( kwiver::vital::feature_set_sptr feat1,
           kwiver::vital::descriptor_set_sptr desc1,
           kwiver::vital::feature_set_sptr feat2,
           kwiver::vital::descriptor_set_sptr desc2 ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::match_set_sptr,
        kwiver::vital::algo::match_features,
        match,
        feat1,
        desc1,
        feat2,
        desc2
      );
    }
};

}
}
}

#endif
