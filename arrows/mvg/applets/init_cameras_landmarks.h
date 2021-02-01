// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_MVG_APPLETS_INIT_CAMERAS_LANDMARKS_H
#define KWIVER_ARROWS_MVG_APPLETS_INIT_CAMERAS_LANDMARKS_H

#include <vital/applets/kwiver_applet.h>
#include <arrows/mvg/applets/kwiver_algo_mvg_applets_export.h>

namespace kwiver {
namespace arrows {
namespace mvg {

class KWIVER_ALGO_MVG_APPLETS_EXPORT init_cameras_landmarks
  : public kwiver::tools::kwiver_applet
{
public:
  init_cameras_landmarks();
  virtual ~init_cameras_landmarks();

  PLUGIN_INFO( "init-cameras-landmarks",
               "Estimate cameras and landmarks from a set of feature tracks");

  virtual int run() override;
  virtual void add_command_options() override;

private:
  class priv;
  std::unique_ptr<priv> d;

}; // end of class

} } } // end namespace

#endif
