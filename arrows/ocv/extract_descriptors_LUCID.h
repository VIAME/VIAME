// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV LUCID descriptor extractor wrapper
 */

#ifndef KWIVER_ARROWS_EXTRACT_DESCRIPTORS_LUCID_H_
#define KWIVER_ARROWS_EXTRACT_DESCRIPTORS_LUCID_H_

#include <opencv2/opencv_modules.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D

#include <memory>
#include <string>

#include <arrows/ocv/extract_descriptors.h>
#include <arrows/ocv/kwiver_algo_ocv_export.h>

namespace kwiver {
namespace arrows {
namespace ocv {

class KWIVER_ALGO_OCV_EXPORT extract_descriptors_LUCID
  : public ocv::extract_descriptors
{
public:
  PLUGIN_INFO( "ocv_LUCID",
               "OpenCV feature-point descriptor extraction via the LUCID algorithm" )

  /// Constructor
  extract_descriptors_LUCID();

  /// Destructor
  virtual ~extract_descriptors_LUCID();

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

private:
  class priv;
  std::unique_ptr<priv> p_;
};

#define KWIVER_OCV_HAS_LUCID

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver

#endif //HAVE_OPENCV_XFEATURES2D

#endif //KWIVER_ARROWS_EXTRACT_DESCRIPTORS_LUCID_H_
