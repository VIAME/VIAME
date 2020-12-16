// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV feature_set implementation
 */

#include "feature_set.h"

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {

/// Return a vector of feature shared pointers
std::vector<feature_sptr>
feature_set
::features() const
{
  typedef std::vector<cv::KeyPoint>::const_iterator cvKP_itr;
  std::vector<feature_sptr> features;
  for(cvKP_itr it = data_.begin(); it != data_.end(); ++it)
  {
    const cv::KeyPoint& kp = *it;
    feature_f *f = new feature_f();
    f->set_loc(vector_2f(kp.pt.x, kp.pt.y));
    f->set_magnitude(kp.response);
    f->set_scale(kp.size);
    f->set_angle(kp.angle);
    features.push_back(feature_sptr(f));
  }
  return features;
}

/// Convert any feature set to a vector of OpenCV cv::KeyPoints
std::vector<cv::KeyPoint>
features_to_ocv_keypoints(const vital::feature_set& feat_set)
{
  if( const ocv::feature_set* f =
          dynamic_cast<const ocv::feature_set*>(&feat_set) )
  {
    return f->ocv_keypoints();
  }
  std::vector<cv::KeyPoint> kpts;
  std::vector<feature_sptr> feat = feat_set.features();
  typedef std::vector<feature_sptr>::const_iterator feat_itr;
  for(feat_itr it = feat.begin(); it != feat.end(); ++it)
  {
    const feature_sptr f = *it;
    cv::KeyPoint kp;
    vector_2d pt = f->loc();
    kp.pt.x = static_cast<float>(pt.x());
    kp.pt.y = static_cast<float>(pt.y());
    kp.response = static_cast<float>(f->magnitude());
    kp.size = static_cast<float>(f->scale());
    kp.angle = static_cast<float>(f->angle());
    kpts.push_back(kp);
  }
  return kpts;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
