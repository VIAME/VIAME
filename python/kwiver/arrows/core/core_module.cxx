#include <pybind11/pybind11.h>

#include <python/kwiver/arrows/core/render_mesh_depth_map.h>
#include <python/kwiver/arrows/core/transfer_bbox_with_depth_map.h>

namespace py = pybind11;

PYBIND11_MODULE(core, m)
{
  render_mesh_depth_map(m);
  transfer_bbox_with_depth_map(m);
}
