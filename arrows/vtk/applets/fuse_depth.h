// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VTK_APPLETS_FUSE_DEPTH_H
#define KWIVER_ARROWS_VTK_APPLETS_FUSE_DEPTH_H

#include <vital/applets/kwiver_applet.h>
#include <arrows/vtk/applets/kwiver_algo_vtk_applets_export.h>

namespace kwiver {
namespace arrows {
namespace vtk {

class KWIVER_ALGO_VTK_APPLETS_EXPORT fuse_depth
  : public kwiver::tools::kwiver_applet
{
public:
  fuse_depth();
  virtual ~fuse_depth();

  PLUGIN_INFO( "fuse-depth",
               "Fuse depth maps from multiple cameras into a single surface");

  int run() override;
  void add_command_options() override;

private:
  class priv;
  std::unique_ptr<priv> d;


}; // end of class

} } } // end namespace

#endif
