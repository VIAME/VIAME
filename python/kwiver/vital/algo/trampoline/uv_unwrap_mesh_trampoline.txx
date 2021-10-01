// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file uv_unwrap_mesh_trampoline.txx
 *
 * \brief trampoline for overriding virtual functions of
 *        algorithm_def<uv_unwrap_mesh> and uv_unwrap_mesh
 */

#ifndef UV_UNWRAP_MESH_TRAMPOLINE_TXX
#define UV_UNWRAP_MESH_TRAMPOLINE_TXX

#include <python/kwiver/vital/util/pybind11.h>
#include <python/kwiver/vital/algo/trampoline/algorithm_trampoline.txx>
#include <vital/algo/uv_unwrap_mesh.h>

namespace kwiver {
namespace vital  {
namespace python {

template < class algorithm_def_uvum_base=
            kwiver::vital::algorithm_def<
              kwiver::vital::algo::uv_unwrap_mesh > >
class algorithm_def_uvum_trampoline :
      public algorithm_trampoline<algorithm_def_uvum_base>
{
  public:
    using algorithm_trampoline<algorithm_def_uvum_base>::algorithm_trampoline;

    std::string type_name() const override
    {
      VITAL_PYBIND11_OVERLOAD(
        std::string,
        kwiver::vital::algorithm_def<kwiver::vital::algo::uv_unwrap_mesh>,
        type_name,
      );
    }
};

template< class uv_unwrap_mesh_base=
                kwiver::vital::algo::uv_unwrap_mesh >
class uv_unwrap_mesh_trampoline :
      public algorithm_def_uvum_trampoline< uv_unwrap_mesh_base >
{
  public:
    using algorithm_def_uvum_trampoline< uv_unwrap_mesh_base>::
              algorithm_def_uvum_trampoline;

    void
    unwrap( kwiver::vital::mesh_sptr mesh ) const override
    {
      VITAL_PYBIND11_OVERLOAD_PURE(
        void,
        kwiver::vital::algo::uv_unwrap_mesh,
        unwrap,
        mesh
      );
    }
};

}
}
}

#endif
