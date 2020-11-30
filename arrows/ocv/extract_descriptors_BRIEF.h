// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV BRIEF descriptor extractor wrapper
 */

#ifndef KWIVER_ARROWS_EXTRACT_DESCRIPTORS_BRIEF_H_
#define KWIVER_ARROWS_EXTRACT_DESCRIPTORS_BRIEF_H_

#include <opencv2/opencv_modules.hpp>
#if KWIVER_OPENCV_VERSION_MAJOR < 3 || defined(HAVE_OPENCV_XFEATURES2D)

#include <vital/algo/extract_descriptors.h>

#include <arrows/ocv/extract_descriptors.h>

#include <string>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT extract_descriptors_BRIEF
  : public extract_descriptors
{
public:
  PLUGIN_INFO( "ocv_BRIEF",
               "OpenCV feature-point descriptor extraction via the BRIEF algorithm" )

  /// Constructor
  extract_descriptors_BRIEF();

  /// Destructor
  virtual ~extract_descriptors_BRIEF();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

private:
  /// private implementation class
  class priv;
  std::unique_ptr<priv> const p_;
};

#define KWIVER_OCV_HAS_BRIEF

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //has OCV support

#endif //KWIVER_ARROWS_EXTRACT_DESCRIPTORS_BRIEF_H_
