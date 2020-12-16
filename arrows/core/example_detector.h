// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_EXAMPLE_DETECTOR_H
#define KWIVER_EXAMPLE_DETECTOR_H

#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/algo/image_object_detector.h>

namespace kwiver {
namespace arrows {
namespace core {

class KWIVER_ALGO_CORE_EXPORT example_detector
        : public vital::algo::image_object_detector
{
public:
  PLUGIN_INFO( "example_detector",
               "Simple example detector that just creates a user-specified bounding box." )

  example_detector();
  virtual ~example_detector();

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config_in);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  // Main detection method
  virtual vital::detected_object_set_sptr detect( vital::image_container_sptr image_data) const;

private:
  class priv;
  const std::unique_ptr<priv> d;
};

} } } // end namespace

#endif //KWIVER_EXAMPLE_DETECTOR_H
