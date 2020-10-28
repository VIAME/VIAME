// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_VTK_APPLETS_ESTIMATE_DEPTH_H
#define KWIVER_ARROWS_VTK_APPLETS_ESTIMATE_DEPTH_H

// TODO
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>

#include <vital/applets/kwiver_applet.h>
#include <arrows/vtk/applets/kwiver_algo_vtk_applets_export.h>

namespace kwiver {
namespace arrows {
namespace vtk {

class KWIVER_ALGO_VTK_APPLETS_EXPORT estimate_depth
  : public kwiver::tools::kwiver_applet
{
public:
  estimate_depth();

  PLUGIN_INFO( "estimate-depth",
               "Depth estimation utility");

  virtual int run() override;
  int run_frame(int frame, const kwiver::vital::camera_map::map_camera_t &cm,
                kwiver::vital::landmark_map_sptr lm, std::string mask_path,
                double angle_span, int num_support, bool hasMask);
  virtual void add_command_options() override;

private:
  class priv;
  std::unique_ptr<priv> d;

}; // end of class

} } } // end namespace

#endif
