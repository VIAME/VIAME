// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file estimate_similarity_transform_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions for
 *        \link kwiver::vital::algo::estimate_similarity_transform \endlink
 */

#ifndef ESTIMATE_SIMILARITY_TRANSFORM_TRAMPOLINE_TXX
#define ESTIMATE_SIMILARITY_TRANSFORM_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/estimate_similarity_transform.h>

namespace kwiver {
namespace vital  {
namespace python {
template< class algorithm_def_est_base=
           kwiver::vital::algorithm_def<
               kwiver::vital::algo::estimate_similarity_transform > >
class algorithm_def_est_trampoline :
      public algorithm_trampoline< algorithm_def_est_base>
{
  public:
    using algorithm_trampoline< algorithm_def_est_base >::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<
              kwiver::vital::algo::estimate_similarity_transform >,
        type_name,
      );
    }
};

template< class estimate_similarity_transform_base =
                  kwiver::vital::algo::estimate_similarity_transform >
class estimate_similarity_transform_trampoline :
      public algorithm_def_est_trampoline< estimate_similarity_transform_base >
{
  public:
    using algorithm_def_est_trampoline< estimate_similarity_transform_base >::
              algorithm_def_est_trampoline;

    kwiver::vital::similarity_d
      estimate_transform( std::vector< kwiver::vital::vector_3d >  const& from,
                          std::vector< kwiver::vital::vector_3d >  const& to
                        )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          std::vector< kwiver::vital::camera_perspective_sptr >  const& from,
          std::vector< kwiver::vital::camera_perspective_sptr >  const& to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          std::vector< kwiver::vital::landmark_sptr >  const& from,
          std::vector< kwiver::vital::landmark_sptr >  const& to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          kwiver::vital::camera_map_sptr const from,
          kwiver::vital::camera_map_sptr const to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }

    kwiver::vital::similarity_d
      estimate_transform(
          kwiver::vital::landmark_map_sptr const from,
          kwiver::vital::landmark_map_sptr const to
          )  const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        kwiver::vital::similarity_d,
        kwiver::vital::algo::estimate_similarity_transform,
        estimate_transform,
        from,
        to
      );
    }
};
}
}
}

#endif
