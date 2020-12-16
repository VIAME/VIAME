// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV match_set implementation
 */

#include "match_set.h"

namespace kwiver {
namespace arrows {
namespace ocv {

/// Return a vector of matching indices
std::vector<vital::match>
match_set
::matches() const
{
  std::vector<vital::match> m;
  for(cv::DMatch dm : this->data_)
  {
    m.push_back( vital::match(dm.queryIdx, dm.trainIdx));
  }
  return m;
}

/// Convert any match set to a vector of OpenCV cv::DMatch
std::vector<cv::DMatch>
matches_to_ocv_dmatch(const vital::match_set& m_set)
{
  if( const ocv::match_set* m_ocv =
          dynamic_cast<const ocv::match_set*>(&m_set) )
  {
    return m_ocv->ocv_matches();
  }
  std::vector<cv::DMatch> dm;
  const std::vector<vital::match> mats = m_set.matches();
  for( vital::match m : mats)
  {
    dm.push_back(cv::DMatch(m.first, m.second, FLT_MAX));
  }
  return dm;
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
