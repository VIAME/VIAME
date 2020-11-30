// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_CORE_TOOLS_RENDER_MESH_H
#define KWIVER_ARROWS_CORE_TOOLS_RENDER_MESH_H

#include <vital/applets/kwiver_applet.h>

#include <arrows/core/applets/kwiver_algo_core_applets_export.h>

#include <string>
#include <vector>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_APPLETS_EXPORT render_mesh
  : public kwiver::tools::kwiver_applet
{
public:
  render_mesh(){}
  virtual ~render_mesh() = default;

  PLUGIN_INFO( "render-mesh",
               "Render a depth or height map from a mesh.\n\n"
               "This tool reads in a mesh file and a camera and renders "
               "various images such as depth map or height map.");

  int run() override;
  void add_command_options() override;

protected:

private:

}; // end of class

} } } // end namespace

#endif /* KWIVER_ARROWS_CORE_TOOLS_RENDER_MESH_H */
