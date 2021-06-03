// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file estimate_fundamental_matrix_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::estimate_fundamental_matrix \endlink
 */

#ifndef ESTIMATE_FUNDAMENTAL_MATRIX_TRAMPOLINE_TXX
#define ESTIMATE_FUNDAMENTAL_MATRIX_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/estimate_fundamental_matrix.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_efm_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::estimate_fundamental_matrix > >
class algorithm_def_efm_trampoline :
      public algorithm_trampoline< algorithm_def_efm_base>
{
  public:
    using algorithm_trampoline< algorithm_def_efm_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
         kwiver::vital::algo::estimate_fundamental_matrix >,
        type_name,
      );
    }
};

template< class estimate_fundamental_matrix_base =
                  kwiver::vital::algo::estimate_fundamental_matrix >
class estimate_fundamental_matrix_trampoline :
      public algorithm_def_efm_trampoline< estimate_fundamental_matrix_base >
{
  public:
    using algorithm_def_efm_trampoline< estimate_fundamental_matrix_base >::
              algorithm_def_efm_trampoline;

    kwiver::vital::fundamental_matrix_sptr
      estimate(const kwiver::vital::feature_set_sptr feat1,
               const kwiver::vital::feature_set_sptr feat2,
               const kwiver::vital::match_set_sptr matches,
               std::vector<bool>& inliers,
               double inlier_scale)  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::fundamental_matrix_sptr,
        kwiver::vital::algo::estimate_fundamental_matrix,
        estimate,
        feat1,
        feat2,
        matches,
        inliers,
        inlier_scale
      );
    }

    kwiver::vital::fundamental_matrix_sptr
      estimate(const std::vector<kwiver::vital::vector_2d>& pts1,
               const std::vector<kwiver::vital::vector_2d>& pts2,
               std::vector<bool>& inliers,
               double inlier_scale)  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::fundamental_matrix_sptr,
        kwiver::vital::algo::estimate_fundamental_matrix,
        estimate,
        pts1,
        pts2,
        inliers,
        inlier_scale
      );
    }

};
}
}
}

#endif
