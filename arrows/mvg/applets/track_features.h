// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_ARROWS_MVG_APPLETS_TRACK_FEATURES_H
#define KWIVER_ARROWS_MVG_APPLETS_TRACK_FEATURES_H

#include <vital/applets/kwiver_applet.h>
#include <arrows/mvg/applets/kwiver_algo_mvg_applets_export.h>

namespace kwiver {
namespace arrows {
namespace mvg {

class KWIVER_ALGO_MVG_APPLETS_EXPORT track_features
  : public kwiver::tools::kwiver_applet
{
public:
  track_features();
  virtual ~track_features();

  PLUGIN_INFO( "track-features",
               "Feature tracking utility");

  int run() override;
  void add_command_options() override;

private:
  class priv;
  const std::unique_ptr<priv> d;

}; // end of class

} } } // end namespace

#endif
